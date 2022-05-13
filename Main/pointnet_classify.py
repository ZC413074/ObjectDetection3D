"""
Author: ZC
Date: May 2022
"""
import importlib
from Common.parse_args import *
from Common.file_operation import *
from Dataset.dataloader import *


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def main(args):
    # step 1: create dirss
    log_dir = create_log_file(args, './log/', 'classification')

    # step 2: data load
    data_path = args.data_path
    train_dataloader, train_dataset = create_dataloader(args, 'train')
    train_dataloader, train_dataset = create_dataloader(args, 'test')

    # step 3: model load
    num_class = args.num_category
    model = importlib.import_module(args.model)  # 动态导入
    classifier = model.PointnetClassifyModel(
        num_class, normal_channel=args.use_normals)
    criterion = model.PointnetClassifyLoss()
    # classifier.apply(inplace_relu)
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # step 4: whether to use premodel
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    # step 5: optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=0.01, momentum=0.9)

    # step 6: lr
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.7)

    # step 7: traning
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        # step7.1 train init
        classifier = classifier.train()
        # step7.2 update lr
        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # step7.3 clear grad
            optimizer.zero_grad()
            points = points.data.numpy()
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            # step7.4 apply model
            pred, trans_feat = classifier(points)
            # step7.5 compute loss
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            # step7.6 compute loss
            loss.backward()
            # step7.7 update  optimizer
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        # step7.8 test
        with torch.no_grad():
            instance_acc, class_acc = test(
                classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' %
                       (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' %
                       (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1


def main_point_net_classify():
    args = point_net_parse_args('train')
    main(args)
