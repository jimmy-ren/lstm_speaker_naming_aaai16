function sn_audio_init()
    global config mem vmem;
    config.input_size = 25;
    config.hidden_layer_size = [512];
    config.output_size = 5;
    config.max_time_steps = 49;
    config.input_valid_len = 49;
    config.output_valid_len = 49;
    config.batch_size = 50;
    config.hidden_layer_num = length(config.hidden_layer_size);

    config.NEW_MEM = @to_gpu;
    % input weights
    config.weights.Whg = {};
    config.weights.Wxg = {};
    % input gate weights
    config.weights.Whi = {};
    config.weights.Wxi = {};
    % output gate weights
    config.weights.Who = {};
    config.weights.Wxo = {};
    % forget gate weights
    config.weights.Whf = {};
    config.weights.Wxf = {};
    
    % bias
    config.weights.Bi = {};
    config.weights.Bf = {};
    config.weights.Bo = {};
    config.weights.Bg = {};
    
    % one layer
    r = 0.08;
    b_offset = 0.3;
    
    % two layers
%     r = 0.28;
%     b_offset = 0.1;
    
    config.r = r;
    config.b_offset = b_offset;
    
    
    config.weights.Whg{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.hidden_layer_size(1))*2*r-r);
    config.weights.Whi{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.hidden_layer_size(1))*2*r-r);
    config.weights.Who{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.hidden_layer_size(1))*2*r-r);
    config.weights.Whf{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.hidden_layer_size(1))*2*r-r);
    config.weights.Wxg{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.input_size)*2*r-r);
    config.weights.Wxi{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.input_size)*2*r-r);
    config.weights.Wxo{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.input_size)*2*r-r);
    config.weights.Wxf{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.input_size)*2*r-r);
%     config.weights.Bi{1} = -0.5;
%     config.weights.Bf{1} = 3;
%     config.weights.Bo{1} = -0.5;
%     config.weights.Bg{1} = 0;
    
    config.weights.Bi{1} = config.NEW_MEM(rand(config.hidden_layer_size(1), 1)) - b_offset;
    config.weights.Bf{1} = config.NEW_MEM(rand(config.hidden_layer_size(1), 1)) + 2 * b_offset;
    config.weights.Bo{1} = config.NEW_MEM(rand(config.hidden_layer_size(1), 1)) - b_offset;
    config.weights.Bg{1} = config.NEW_MEM(rand(config.hidden_layer_size(1), 1));
    for L = 2:config.hidden_layer_num
        config.weights.Whg{L} = config.NEW_MEM(randn(config.hidden_layer_size(L),config.hidden_layer_size(L))*r);
        config.weights.Whi{L} = config.NEW_MEM(randn(config.hidden_layer_size(L),config.hidden_layer_size(L))*r);
        config.weights.Who{L} = config.NEW_MEM(randn(config.hidden_layer_size(L),config.hidden_layer_size(L))*r);
        config.weights.Whf{L} = config.NEW_MEM(randn(config.hidden_layer_size(L),config.hidden_layer_size(L))*r);
        config.weights.Wxg{L} = config.NEW_MEM(randn(config.hidden_layer_size(L),config.hidden_layer_size(L-1))*r);
        config.weights.Wxi{L} = config.NEW_MEM(randn(config.hidden_layer_size(L),config.hidden_layer_size(L-1))*r);
        config.weights.Wxo{L} = config.NEW_MEM(randn(config.hidden_layer_size(L),config.hidden_layer_size(L-1))*r);
        config.weights.Wxf{L} = config.NEW_MEM(randn(config.hidden_layer_size(L),config.hidden_layer_size(L-1))*r);
        
        config.weights.Bi{L} = config.NEW_MEM(rand(config.hidden_layer_size(L), 1)) - b_offset;
        config.weights.Bf{L} = config.NEW_MEM(rand(config.hidden_layer_size(L), 1)) + 2 * b_offset;
        config.weights.Bo{L} = config.NEW_MEM(rand(config.hidden_layer_size(L), 1)) - b_offset;
        config.weights.Bg{L} = config.NEW_MEM(rand(config.hidden_layer_size(L), 1));
    end
    % output weights
    config.weights.Wy = config.NEW_MEM(rand(config.output_size,config.hidden_layer_size(1))*2*r-r);    
    
    
    mem.cell_in = {};
    mem.in_gate = {};
    mem.out_gate = {};
    mem.forget_gate = {};
    mem.cell_state = {};
    mem.cell_acts = {}; % activation of the lstm unit
    mem.net_out = 0;
    
    % for each layer
    for L = 1:config.hidden_layer_num
        mem.cell_in{L} = config.NEW_MEM(zeros(size(config.weights.Whg{L},1), config.batch_size, config.max_time_steps));
        mem.in_gate{L} = config.NEW_MEM(zeros(size(mem.cell_in{L})));
        mem.out_gate{L} = config.NEW_MEM(zeros(size(mem.cell_in{L})));
        mem.forget_gate{L} = config.NEW_MEM(zeros(size(mem.cell_in{L})));
        mem.cell_state{L} = config.NEW_MEM(zeros(size(mem.cell_in{L})));
        mem.cell_acts{L} = config.NEW_MEM(zeros(size(mem.cell_in{L})));
    end
    
    
    vmem.cell_in = {};
    vmem.in_gate = {};
    vmem.out_gate = {};
    vmem.forget_gate = {};
    vmem.cell_state = {};
    vmem.cell_acts = {}; % activation of the lstm unit
    vmem.net_out = 0;
    
    for L = 1:config.hidden_layer_num
        vmem.cell_in{L} = config.NEW_MEM(zeros(size(config.weights.Whg{L},1), 1, config.max_time_steps));
        vmem.in_gate{L} = config.NEW_MEM(zeros(size(vmem.cell_in{L})));
        vmem.out_gate{L} = config.NEW_MEM(zeros(size(vmem.cell_in{L})));
        vmem.forget_gate{L} = config.NEW_MEM(zeros(size(vmem.cell_in{L})));
        vmem.cell_state{L} = config.NEW_MEM(zeros(size(vmem.cell_in{L})));
        vmem.cell_acts{L} = config.NEW_MEM(zeros(size(vmem.cell_in{L})));
    end
    
    % for training only
    mem.delta_x = {};
    mem.delta_h_prev = {};
    mem.partial_h_ratio = {};
    for L = 1:config.hidden_layer_num
        if(L > 1)
            mem.delta_x{L} = config.NEW_MEM(zeros(config.hidden_layer_size(L-1), config.batch_size, config.max_time_steps));
        end
        mem.delta_h_prev{L} = config.NEW_MEM(zeros(config.hidden_layer_size(L), config.batch_size, config.max_time_steps+1));
        mem.partial_h_ratio{L} = config.NEW_MEM(ones(config.hidden_layer_size(L), config.batch_size, config.max_time_steps+1));
    end
    mem.delta_x{L+1} = config.NEW_MEM(zeros(config.hidden_layer_size(L), config.batch_size, config.max_time_steps)); % before output
    
    
    % input gradients
    mem.grad.Whg = {};
    mem.grad.Wxg = {};
    % input gate gradients
    mem.grad.Whi = {};
    mem.grad.Wxi = {};
    % output gate gradients
    mem.grad.Who = {};
    mem.grad.Wxo = {};
    % forget gate gradients
    mem.grad.Whf = {};
    mem.grad.Wxf = {};
    
    % bias gradients
    mem.grad.Bi = {};
    mem.grad.Bf = {};
    mem.grad.Bo = {};
    mem.grad.Bg = {};
    
    for L = 1:config.hidden_layer_num
        mem.grad.Whg{L} = config.NEW_MEM(zeros(size(config.weights.Whg{L})));
        mem.grad.Wxg{L} = config.NEW_MEM(zeros(size(config.weights.Wxg{L})));
        mem.grad.Whi{L} = config.NEW_MEM(zeros(size(config.weights.Whi{L})));
        mem.grad.Wxi{L} = config.NEW_MEM(zeros(size(config.weights.Wxi{L})));
        mem.grad.Who{L} = config.NEW_MEM(zeros(size(config.weights.Who{L})));
        mem.grad.Wxo{L} = config.NEW_MEM(zeros(size(config.weights.Wxo{L})));
        mem.grad.Whf{L} = config.NEW_MEM(zeros(size(config.weights.Whf{L})));
        mem.grad.Wxf{L} = config.NEW_MEM(zeros(size(config.weights.Wxf{L})));
        
        mem.grad.Bi{L} = config.NEW_MEM(zeros(config.hidden_layer_size(L), 1));
        mem.grad.Bf{L} = config.NEW_MEM(zeros(config.hidden_layer_size(L), 1));
        mem.grad.Bo{L} = config.NEW_MEM(zeros(config.hidden_layer_size(L), 1));
        mem.grad.Bg{L} = config.NEW_MEM(zeros(config.hidden_layer_size(L), 1));
    end    
end





