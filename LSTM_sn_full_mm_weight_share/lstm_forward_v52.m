function lstm_forward_v52(in, label)
    global config mem;
    batch_size = config.batch_size;
    gen_net_out();
    % forward pass
    for L = 1:config.hidden_layer_num    % for each hidden layer
        for s = 1:config.slide_pieces
            x = get_x(L, s, in);
            if(s == 1)
                cell_in_in = pagefun(@mtimes, repmat(config.weights.Wxg{L}{s}*0.8, [1 1 size(in,2)]), x);
                in_gate_in = pagefun(@mtimes, repmat(config.weights.Wxi{L}{s}*0.8, [1 1 size(in,2)]), x);
                forget_gate_in = pagefun(@mtimes, repmat(config.weights.Wxf{L}{s}*0.8, [1 1 size(in,2)]), x);
                out_gate_in = pagefun(@mtimes, repmat(config.weights.Wxo{L}{s}*0.8, [1 1 size(in,2)]), x);
            else
                cell_in_in = pagefun(@mtimes, repmat(config.weights.Wxg{L}{s}*0.8, [1 1 size(in,2)]), x);
                in_gate_in = pagefun(@mtimes, repmat(config.weights.Wxi{L}{s}*0.8, [1 1 size(in,2)]), x);
                forget_gate_in = pagefun(@mtimes, repmat(config.weights.Wxf{L}{s}*0.8, [1 1 size(in,2)]), x);
                out_gate_in = pagefun(@mtimes, repmat(config.weights.Wxo{L}{s}*0.8, [1 1 size(in,2)]), x);
            end
            for t = 1:size(in,2)    % for each time step
                [hprev, csprev] = get_hprev_csprev(t, L, s);
                mem.cell_in{L}{s}(:,:,t) = tanh(bsxfun(@plus, config.weights.Whg{L} * hprev + cell_in_in(:,:,t), config.weights.Bg{L}{s}));
                mem.in_gate{L}{s}(:,:,t) = sigmoid(bsxfun(@plus, config.weights.Whi{L} * hprev + in_gate_in(:,:,t), config.weights.Bi{L}{s}));
                mem.forget_gate{L}{s}(:,:,t) = sigmoid(bsxfun(@plus, config.weights.Whf{L} * hprev + forget_gate_in(:,:,t), config.weights.Bf{L}{s}));
                mem.out_gate{L}{s}(:,:,t) = sigmoid(bsxfun(@plus, config.weights.Who{L} * hprev + out_gate_in(:,:,t), config.weights.Bo{L}{s}));

                mem.cell_state{L}{s}(:,:,t) = mem.cell_in{L}{s}(:,:,t) .*  mem.in_gate{L}{s}(:,:,t) + csprev .* mem.forget_gate{L}{s}(:,:,t);
                mem.cell_acts{L}{s}(:,:,t) = tanh(mem.cell_state{L}{s}(:,:,t)) .* mem.out_gate{L}{s}(:,:,t);
            end
        end
    end
    
    %dm = config.NEW_MEM(uint8(rand(size(mem.cell_acts{L}))));
    %mem.cell_acts{L} = mem.cell_acts{L} .* dm;
    
    label(:,1:size(in,2)-config.output_valid_len,:) = 0;
    
    cost_arr = cell(1, config.slide_pieces);
    config.cost = {};
    
    % softmax + cross entropy
    for s = 1:config.slide_pieces
        mem.net_out(:,size(in,2)-config.output_valid_len+1:end,:,s) = permute(softmax(pagefun(@mtimes, repmat(config.weights.Wy, [1 1 config.output_valid_len]), mem.cell_acts{config.hidden_layer_num}{s}(:,:,size(in,2)-config.output_valid_len+1:end))), [1 3 2]);
        cost_arr{s} = sum(-sum(label(:,size(in,2)-config.output_valid_len+1:end,:) .* log(mem.net_out(:,size(in,2)-config.output_valid_len+1:end,:,s))), 3) / batch_size;
    
        config.cost{s} = sum(cost_arr{s}); % overall cost
    end



end



function [hprev, csprev] = get_hprev_csprev(t, L, s)
    % csprev: previous cell state
    global config mem;
%     if(L == 1)
%         x = reshape(in(:,t,:), size(in,1), size(in,3));  % to make the whole batch of samples at time step t a 2D matrix
%     else
%         x = mem.cell_acts{L-1}(:,:,t);
%     end
    if(t == 1)
        hprev = config.NEW_MEM(zeros(config.hidden_layer_size(L), config.batch_size));
        csprev = 0;
    else
        hprev = mem.cell_acts{L}{s}(:,:,t-1);
        csprev = mem.cell_state{L}{s}(:,:,t-1);
    end
end

function x = get_x(L, s, in)
    global config mem;
    if(L == 1)
        x = permute(in(config.slide_endpoints(s)+1:config.slide_endpoints(s+1),:,:), [1 3 2]);
    else
        x = mem.cell_acts{L-1}{s};
    end
end

function gen_net_out()
    global config mem;
    if(size(mem.net_out,2) == 1)
        mem.net_out = config.NEW_MEM(zeros(config.output_size, config.max_time_steps, config.batch_size, config.slide_pieces));
    end
end
