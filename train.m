function [W1, W3, W4] = train(W1, W3, W4, data, tag, lr)
    sz_data = size(data);
    sz_tag = size(tag);
    sz_W1 = size(W1);
    sz_filter = sz_data(1) - sz_W1(1) + 1;
    
    all_dW1 = zeros(size(W1));
    all_dW3 = zeros(size(W3));
    all_dW4 = zeros(size(W4));
    V1 = zeros(sz_filter, sz_filter, sz_W1(3));
    dW1 = zeros(size(W1));
    
    data = reshape(data, [sz_data(1), sz_data(2), sz_data(4)]);
    tag = reshape(tag, [sz_tag(1), sz_tag(3)]);
    
    for i = 1:sz_data(4)
        for j = 1:sz_W1(3)
            V1(:, :, j) = filter2(W1(:, :, j), data(:, :, i), 'valid');
        end
        Y1 = max(0, V1);
        Y2 = (Y1(1:2:end, 1:2:end, :) + Y1(2:2:end, 1:2:end, :) +...
            Y1(1:2:end, 2:2:end, :) + Y1(2:2:end, 2:2:end, :)) / 4;
        y2 = reshape(Y2, [], 1);
        v3 = W3 * y2;
        y3 = max(0, v3);
        v = W4 * y3;
        y = Softmax(v);
        
        e = tag(:, i) - y;
        delta = e;
        dW4 = lr * delta * y3';
        e3  = W4' * delta;
        delta3 = (v3 > 0) .* e3;
        dW3 = lr * delta3 * y2';
        e2 = W3' * delta3;
        E2 = reshape(e2, size(Y2));
        E2_4 = E2 / 4;
        E1 = zeros(size(Y1));
        E1(1:2:end, 1:2:end, :) = E2_4;
        E1(1:2:end, 2:2:end, :) = E2_4;
        E1(2:2:end, 1:2:end, :) = E2_4;
        E1(2:2:end, 2:2:end, :) = E2_4;
        delta1 = (V1 > 0) .* E1;
        for j = 1:length(W1)
            dW1(:, :, j) = lr * filter2(delta1(:, :, j), data(:, :, i), 'valid');
        end
        
        all_dW1 = all_dW1 + dW1;
        all_dW3 = all_dW3 + dW3;
        all_dW4 = all_dW4 + dW4;
    end

    dW1 = all_dW1 / sz_data(4);
    dW3 = all_dW3 / sz_data(4);
    dW4 = all_dW4 / sz_data(4);
    W1 = W1 + dW1;
    W3 = W3 + dW3;
    W4 = W4 + dW4;

    