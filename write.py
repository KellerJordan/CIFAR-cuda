import airbench
loader = airbench.CifarLoader('/tmp/cifar10', train=False)
loader.normalize(loader.images).cpu().float().numpy().tofile('test_x.bin')
loader.labels.cpu().long().numpy().tofile('test_y.bin')
loader = airbench.CifarLoader('/tmp/cifar10', train=True)
loader.normalize(loader.images).cpu().float().numpy().tofile('train_x.bin')
loader.labels.cpu().long().numpy().tofile('train_y.bin')
