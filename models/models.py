
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'recycle_gan':
        assert(opt.dataset_mode == 'unaligned_triplet')
        from .recycle_gan_model import RecycleGANModel
        model = RecycleGANModel()
    elif opt.model == 'reCycle_gan':
        assert(opt.dataset_mode == 'unaligned_triplet')
        from .reCycle_gan_model import ReCycleGANModel
        model = ReCycleGANModel()
    elif opt.model == 'fullrecycle_gan':
        assert(opt.dataset_mode == 'unaligned_triplet')
        from .fullrecycle_gan_model import FullRecycleGANModel
        model = FullRecycleGANModel()
    elif opt.model == 'revrecycle_gan':
        assert(opt.dataset_mode == 'unaligned_triplet')
        from .revrecycle_gan_model import RevRecycleGANModel
        model = RevRecycleGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
