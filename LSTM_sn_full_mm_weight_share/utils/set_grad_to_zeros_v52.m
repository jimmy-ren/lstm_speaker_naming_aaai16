function set_grad_to_zeros_v52()
    global config mem;
    %z = config.NEW_MEM(0);
    for L = 1:config.hidden_layer_num
        mem.grad.Whg{L} = config.NEW_MEM(zeros(size(config.weights.Whg{L})));
        mem.grad.Whi{L} = config.NEW_MEM(zeros(size(config.weights.Whi{L})));
        mem.grad.Who{L} = config.NEW_MEM(zeros(size(config.weights.Who{L})));
        mem.grad.Whf{L} = config.NEW_MEM(zeros(size(config.weights.Whf{L})));
        for s = 1:config.slide_pieces
            mem.grad.Wxg{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxg{L}{s})));
            mem.grad.Wxi{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxi{L}{s})));
            mem.grad.Wxo{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxo{L}{s})));
            mem.grad.Wxf{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxf{L}{s})));
            mem.grad.Bi{L}{s} = config.NEW_MEM(zeros(size(config.weights.Bi{L}{s})));
            mem.grad.Bf{L}{s} = config.NEW_MEM(zeros(size(config.weights.Bf{L}{s})));
            mem.grad.Bo{L}{s} = config.NEW_MEM(zeros(size(config.weights.Bo{L}{s})));
            mem.grad.Bg{L}{s} = config.NEW_MEM(zeros(size(config.weights.Bg{L}{s})));
        end
    end
    mem.grad.Wy = config.NEW_MEM(zeros(size(config.weights.Wy)));
end



