"""
EEG-MVSelect 학습 및 평가 모듈
"""

import time
import torch
import torch.nn.functional as F
from src.models.mvselect import aggregate_feat, get_eps_thres


class EEGTrainer:
    """EEG-MVSelect 학습 및 평가를 위한 Trainer 클래스"""
    
    def __init__(self, model, args):
        self.model = model
        self.args = args
        
    def train_epoch(self, epoch, dataloader, optimizer, scheduler=None, log_interval=50):
        """한 에폭 학습"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # MVSelect 사용시 추가 통계
        if self.args.steps > 0:
            action_sum = torch.zeros(self.model.num_cam).cuda()
            
        for batch_idx, (imgs, labels, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            imgs = imgs.cuda()
            labels = labels.cuda()
            keep_cams = keep_cams.cuda()
            
            optimizer.zero_grad()
            
            if self.args.steps == 0:
                # 모든 뷰 사용
                output, _, _ = self.model(imgs, down=self.args.down)
                loss = F.cross_entropy(output, labels)
                
                pred = output.argmax(dim=1)
                correct += (pred == labels).sum().item()
                
            else:
                # MVSelect 사용
                feat, _ = self.model.get_feat(imgs, down=self.args.down, visualize=True)
                
                # 초기 뷰 선택 (랜덤 또는 첫 번째 뷰)
                if self.args.init_cam == 'random':
                    init_cam = torch.randint(0, N, (B,))
                else:
                    init_cam = torch.zeros(B, dtype=torch.long)
                init_prob = F.one_hot(init_cam, num_classes=N).cuda()
                
                # Epsilon-greedy 파라미터
                eps_thres = get_eps_thres(epoch, self.args.epochs)
                
                # 순차적 뷰 선택
                overall_feat, (log_probs, values, actions, entropies) = \
                    self.model.do_steps(feat, init_prob, self.args.steps, keep_cams)
                
                # 출력 및 손실
                output = self.model.get_output(overall_feat)
                ce_loss = F.cross_entropy(output, labels)
                
                # 강화학습 손실 (DQN 스타일)
                if self.args.rl_loss_weight > 0:
                    # Reward: 정확한 예측 시 1, 틀린 예측 시 0
                    pred = output.argmax(dim=1)
                    rewards = (pred == labels).float()
                    
                    # Q-learning loss
                    rl_loss = 0
                    for step_values, action in zip(values, actions):
                        selected_values = (step_values * action.float()).sum(dim=1)
                        rl_loss += F.mse_loss(selected_values, rewards.detach())
                    rl_loss = rl_loss / len(values)
                    
                    loss = ce_loss + self.args.rl_loss_weight * rl_loss
                else:
                    loss = ce_loss
                
                # 통계
                correct += (pred == labels).sum().item()
                for action in actions:
                    action_sum += action.sum(dim=0)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            total += B
            
            # 스케줄러 업데이트
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler.step(epoch + batch_idx / len(dataloader))
            
            # 로깅
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100.0 * correct / total
                
                log_str = f'Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] ' \
                         f'Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%'
                
                if self.args.steps > 0:
                    # 선택된 뷰 빈도
                    view_freq = F.normalize(action_sum, p=1, dim=0)
                    top_views = torch.topk(view_freq, k=min(3, len(view_freq)))
                    log_str += f' | Top views: '
                    for idx, freq in zip(top_views.indices, top_views.values):
                        log_str += f'{idx.item()}({freq.item()*100:.1f}%) '
                
                print(log_str)
        
        # 에폭 평균
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader, init_cam_list=None):
        """평가"""
        self.model.eval()
        
        # init_cam_list: MVSelect 평가시 여러 초기 뷰로 테스트
        if init_cam_list is None:
            init_cam_list = [None]
        
        K = len(init_cam_list)
        losses = torch.zeros(K)
        correct = torch.zeros(K)
        total = torch.zeros(K)
        
        if self.args.steps > 0:
            action_sum = torch.zeros(K, self.model.num_cam).cuda()
        
        t0 = time.time()
        
        with torch.no_grad():
            for imgs, labels, keep_cams in dataloader:
                B, N = imgs.shape[:2]
                imgs = imgs.cuda()
                labels = labels.cuda()
                keep_cams = keep_cams.cuda()
                
                for k, init_cam in enumerate(init_cam_list):
                    if self.args.steps == 0 or init_cam is None:
                        # 모든 뷰 사용
                        output, _, _ = self.model(imgs, down=self.args.down)
                        actions_k = []
                    else:
                        # MVSelect 사용
                        feat, _ = self.model.get_feat(imgs, down=self.args.down)
                        init_prob = F.one_hot(torch.tensor(init_cam).repeat(B), num_classes=N).cuda()
                        
                        overall_feat, (_, _, actions_k, _) = \
                            self.model.do_steps(feat, init_prob, self.args.steps, keep_cams)
                        output = self.model.get_output(overall_feat)
                        
                        # 액션 통계
                        for action in actions_k:
                            action_sum[k] += action.sum(dim=0)
                    
                    # 손실 및 정확도
                    loss = F.cross_entropy(output, labels)
                    losses[k] += loss.item()
                    
                    pred = output.argmax(dim=1)
                    correct[k] += (pred == labels).sum().item()
                    total[k] += B
        
        # 결과 출력
        print('\n' + '='*70)
        for k in range(K):
            avg_loss = losses[k] / len(dataloader)
            accuracy = 100.0 * correct[k] / total[k]
            
            if init_cam_list[k] is not None:
                print(f'Init view {init_cam_list[k]}: Loss {avg_loss:.4f}, Acc {accuracy:.2f}%')
                
                if self.args.steps > 0:
                    # 선택된 뷰 분포
                    view_freq = F.normalize(action_sum[k], p=1, dim=0).cpu()
                    print(f'  Selected views: ', end='')
                    for view_idx, freq in enumerate(view_freq):
                        if freq > 0.01:  # 1% 이상만 출력
                            print(f'{view_idx}({freq*100:.1f}%) ', end='')
                    print()
            else:
                print(f'All views: Loss {avg_loss:.4f}, Acc {accuracy:.2f}%')
        
        if K > 1 and init_cam_list[0] is not None:
            # MVSelect 평균 성능
            avg_accuracy = (correct / total).mean() * 100
            std_accuracy = (correct / total).std() * 100
            print(f'\nMVSelect Average: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%')
        
        print(f'Time: {time.time() - t0:.1f}s')
        print('='*70 + '\n')
        
        return losses / len(dataloader), correct / total * 100
    
    def test_oracle(self, dataloader):
        """
        Oracle 성능 측정: 모든 가능한 뷰 조합을 테스트하여
        최적의 조합 찾기 (상한 성능)
        """
        import itertools
        
        self.model.eval()
        N = self.model.num_cam
        
        # 테스트할 조합 (steps개의 뷰 선택)
        if self.args.steps == 0:
            combinations = [tuple(range(N))]
        else:
            combinations = list(itertools.combinations(range(N), self.args.steps + 1))
        
        print(f'\nOracle Test: Testing {len(combinations)} combinations...')
        
        combination_scores = torch.zeros(len(combinations))
        
        with torch.no_grad():
            for imgs, labels, keep_cams in dataloader:
                B, N_view = imgs.shape[:2]
                imgs = imgs.cuda()
                labels = labels.cuda()
                
                feat, _ = self.model.get_feat(imgs, down=self.args.down)
                
                for comb_idx, comb in enumerate(combinations):
                    # 해당 조합의 뷰만 선택
                    selection = torch.zeros(B, N_view).cuda()
                    for view_idx in comb:
                        selection[:, view_idx] = 1
                    
                    # Forward
                    overall_feat = aggregate_feat(feat, selection, self.model.aggregation)
                    output = self.model.get_output(overall_feat)
                    
                    # 정확도
                    pred = output.argmax(dim=1)
                    correct = (pred == labels).sum().item()
                    combination_scores[comb_idx] += correct
        
        # 최고 조합 찾기
        best_comb_idx = combination_scores.argmax().item()
        best_comb = combinations[best_comb_idx]
        best_accuracy = combination_scores[best_comb_idx] / len(dataloader.dataset) * 100
        
        print(f'Best combination: {best_comb}')
        print(f'Oracle accuracy: {best_accuracy:.2f}%')
        print()
        
        return best_comb, best_accuracy


def create_optimizer(model, args):
    """옵티마이저 생성"""
    if args.steps == 0:
        # MVSelect 없이 전체 모델 학습
        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        # MVSelect 사용: base와 select 모듈 분리 학습
        base_params = []
        select_params = []
        
        for name, param in model.named_parameters():
            if 'select_module' in name:
                select_params.append(param)
            else:
                base_params.append(param)
        
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': args.lr * args.base_lr_ratio},
            {'params': select_params, 'lr': args.select_lr}
        ], weight_decay=args.weight_decay)
    
    return optimizer


def create_scheduler(optimizer, args, steps_per_epoch):
    """스케줄러 생성"""
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.1
        )
    elif args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, 
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs
        )
    else:
        scheduler = None
    
    return scheduler
