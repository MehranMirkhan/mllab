
# Layer
- torch.nn.Linear(in_features, out_features, bias=True)
- torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
- torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
- torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
- torch.nn.MaxUnpool2d(kernel_size, stride=None, padding=0)
- torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
- torch.nn.Dropout2d(p=0.5, inplace=False)
- torch.nn.RNN
- torch.nn.LSTM

# Activation
- torch.relu
- torch.sigmoid
- torch.tanh
- torch.nn.Hardtanh
- torch.nn.LeakyReLU
- torch.nn.Softplus
- torch.nn.Softmax
- torch.nn.ELU

# Loss
- torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
- torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
- torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

# Optimizer
- torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
- torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
- torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
- torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
- torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
- torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
- torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
- torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
- torch.optim.SGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
## Adjust learning rate
- torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
- torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
- torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
- torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# Save & Load
- torch.save(model.state_dict(), PATH)
- model.load_state_dict(torch.load(PATH))
- model.eval()      [Must be called after loading]
