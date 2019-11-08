function [loss, precision] = evaluation(W1, W3, W4, data, tag)  
    sz_data = size(data);
    sz_W1 = size(W1);
    sz_filter = sz_data(1) - sz_W1(1) + 1;
    V1 = zeros(sz_filter, sz_filter, sz_W1(3));
    all_loss = 0;
    count = 0;
    for i = 1:sz_data(3)
        predict = zeros(10, 1);
        for j = 1:length(W1)  
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
        [~, idx] = max(y);
        predict(idx) = 1;
        loss = CrossEntropy(y, tag(:, i));
        all_loss  = all_loss + loss;
        if predict == tag(:, i)
            count = count + 1;
        end
    end
    loss = all_loss / length(data);
    precision = count / length(data);