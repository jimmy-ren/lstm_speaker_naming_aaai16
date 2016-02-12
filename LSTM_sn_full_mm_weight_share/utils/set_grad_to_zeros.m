function set_grad_to_zeros()
    global config mem;    
    for L = 1:config.hidden_layer_num
        mem.grad.Whg{L} = config.NEW_MEM(zeros(size(config.weights.Whg{L})));
        mem.grad.Whi{L} = config.NEW_MEM(zeros(size(config.weights.Whi{L})));
        mem.grad.Who{L} = config.NEW_MEM(zeros(size(config.weights.Who{L})));
        mem.grad.Whf{L} = config.NEW_MEM(zeros(size(config.weights.Whf{L})));
        
        mem.grad.Wxg{L} = config.NEW_MEM(zeros(size(config.weights.Wxg{L})));
        mem.grad.Wxi{L} = config.NEW_MEM(zeros(size(config.weights.Wxi{L})));
        mem.grad.Wxo{L} = config.NEW_MEM(zeros(size(config.weights.Wxo{L})));
        mem.grad.Wxf{L} = config.NEW_MEM(zeros(size(config.weights.Wxf{L})));
        mem.grad.Bi{L} = config.NEW_MEM(zeros(size(config.weights.Bi{L})));
        mem.grad.Bf{L} = config.NEW_MEM(zeros(size(config.weights.Bf{L})));
        mem.grad.Bo{L} = config.NEW_MEM(zeros(size(config.weights.Bo{L})));
        mem.grad.Bg{L} = config.NEW_MEM(zeros(size(config.weights.Bg{L})));
    end
    mem.grad.Wy = config.NEW_MEM(zeros(size(config.weights.Wy)));
end


