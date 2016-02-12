function set_grad_to_zeros_v6()
    global config mem;
    %z = config.NEW_MEM(0);
    for L = 1:config.hidden_layer_num        
        for s = 1:config.slide_pieces
            mem.grad.Whg{L}{s} = config.NEW_MEM(zeros(size(config.weights.Whg{L}{s})));
            mem.grad.Whi{L}{s} = config.NEW_MEM(zeros(size(config.weights.Whi{L}{s})));
            mem.grad.Who{L}{s} = config.NEW_MEM(zeros(size(config.weights.Who{L}{s})));
            mem.grad.Whf{L}{s} = config.NEW_MEM(zeros(size(config.weights.Whf{L}{s})));
            
            mem.grad.Wxg{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxg{L}{s})));
            mem.grad.Wxi{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxi{L}{s})));
            mem.grad.Wxo{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxo{L}{s})));
            mem.grad.Wxf{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxf{L}{s})));
            mem.grad.Bi{L}{s} = config.NEW_MEM(zeros(size(config.weights.Bi{L}{s})));
            mem.grad.Bf{L}{s} = config.NEW_MEM(zeros(size(config.weights.Bf{L}{s})));
            mem.grad.Bo{L}{s} = config.NEW_MEM(zeros(size(config.weights.Bo{L}{s})));
            mem.grad.Bg{L}{s} = config.NEW_MEM(zeros(size(config.weights.Bg{L}{s})));
            
            mem.grad.Wy{s} = config.NEW_MEM(zeros(size(config.weights.Wy{s})));
        end
    end
    %mem.grad.Wy = config.NEW_MEM(zeros(size(config.weights.Wy)));
end


