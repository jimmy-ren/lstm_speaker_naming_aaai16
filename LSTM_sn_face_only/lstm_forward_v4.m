function lstm_forward_v4(in, label)
    % in is a n by t by b matrix, where n is the input data dimension
    % t is the number of input samples (time steps)
    % b is batch size    
    
    global config mem;
    batch_size = config.batch_size;
    gen_net_out();
    % forward pass
    for L = 1:config.hidden_layer_num    % for each hidden layer
        x = get_x(L, in);
        cell_in_in = pagefun(@mtimes, repmat(config.weights.Wxg{L}, [1 1 size(in,2)]), x);
        in_gate_in = pagefun(@mtimes, repmat(config.weights.Wxi{L}, [1 1 size(in,2)]), x);
        forget_gate_in = pagefun(@mtimes, repmat(config.weights.Wxf{L}, [1 1 size(in,2)]), x);
        out_gate_in = pagefun(@mtimes, repmat(config.weights.Wxo{L}, [1 1 size(in,2)]), x);
        for t = 1:size(in,2)    % for each time step
            [x, hprev, csprev] = get_x_hprev_csprev(in, t, L);
            mem.cell_in{L}(:,:,t) = tanh(bsxfun(@plus, config.weights.Whg{L} * hprev + cell_in_in(:,:,t), config.weights.Bg{L}));
            mem.in_gate{L}(:,:,t) = sigmoid(bsxfun(@plus, config.weights.Whi{L} * hprev + in_gate_in(:,:,t), config.weights.Bi{L}));
            mem.forget_gate{L}(:,:,t) = sigmoid(bsxfun(@plus, config.weights.Whf{L} * hprev + forget_gate_in(:,:,t), config.weights.Bf{L}));
            mem.out_gate{L}(:,:,t) = sigmoid(bsxfun(@plus, config.weights.Who{L} * hprev + out_gate_in(:,:,t), config.weights.Bo{L}));
            
            mem.cell_state{L}(:,:,t) = mem.cell_in{L}(:,:,t) .*  mem.in_gate{L}(:,:,t) + csprev .* mem.forget_gate{L}(:,:,t);
            mem.cell_acts{L}(:,:,t) = tanh(mem.cell_state{L}(:,:,t)) .* mem.out_gate{L}(:,:,t);
        end
    end
    
    %mem.cell_acts{L} = mem.cell_acts{L} ./ 2;
    
    % softmax + cross entropy
    mem.net_out(:,size(in,2)-config.output_valid_len+1:end,:) = permute(softmax(pagefun(@mtimes, repmat(config.weights.Wy, [1 1 config.output_valid_len]), mem.cell_acts{config.hidden_layer_num}(:,:,size(in,2)-config.output_valid_len+1:end))), [1 3 2]);
    cost_arr = sum(-sum(label(:,size(in,2)-config.output_valid_len+1:end,:) .* log(mem.net_out(:,size(in,2)-config.output_valid_len+1:end,:))), 3) / batch_size;
    
    % tanh + L2 norm
    %mem.net_out(:,size(in,2)-config.output_valid_len+1:end,:) = permute(tanh(pagefun(@mtimes, repmat(config.weights.Wy, [1 1 config.output_valid_len]), mem.cell_acts{config.hidden_layer_num}(:,:,size(in,2)-config.output_valid_len+1:end))), [1 3 2]);
    %cost_arr = sum(sum((mem.net_out(:,size(in,2)-config.output_valid_len+1:end,:) - label(:,size(in,2)-config.output_valid_len+1:end,:)).^2, 3)) / 2 / batch_size;    
    
    config.cost = sum(cost_arr); % overall cost
end

function [x, hprev, csprev] = get_x_hprev_csprev(in, t, L)
    % csprev: previous cell state
    global config mem;
    if(L == 1)
        x = reshape(in(:,t,:), size(in,1), size(in,3));  % to make the whole batch of samples at time step t a 2D matrix
    else
        x = mem.cell_acts{L-1}(:,:,t);
    end
    if(t == 1)
        hprev = config.NEW_MEM(zeros(config.hidden_layer_size(L), config.batch_size));
        csprev = 0;
    else
        hprev = mem.cell_acts{L}(:,:,t-1);
        csprev = mem.cell_state{L}(:,:,t-1);
    end
end

function x = get_x(L, in)
    global mem;
    if(L == 1)
        x = permute(in, [1 3 2]);
    else
        x = mem.cell_acts{L-1};
    end
end

function gen_net_out()
    global config mem;
    if(size(mem.net_out,2) == 1)
        mem.net_out = config.NEW_MEM(zeros(config.output_size, config.max_time_steps, config.batch_size));
    end
end





