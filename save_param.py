from config import Config as cfg

def write_options(file_name, optimizer_name, lr, weight_decay, end_epoch, train_state):
    file_name = file_name + '.txt'
    img_resize_list = 'img_resize_list : ' + str(cfg.Lidar_set.resize_list)+'\n'
    optimizer_ = 'optimizer : ' + optimizer_name + ' , lr :' + str(lr) + ', weight decay : ' + str(weight_decay)+'\n'
    epoch = 'train epoch : ' + str(end_epoch)+'\n'
    model = 'model : vgg16 \n'
    if train_state:
        re_train = 'Train state : ReTrain model'
    else:
        re_train = 'Train state : First Train model'
    with open(file_name, 'w') as f:
        f.write(img_resize_list)
        f.write(optimizer_)
        f.write(epoch)
        f.write(model)
        f.write(re_train)