epochs = 3;
batch_size = 60;
lr = 0.15;
Data_Processor();
train_data_path = "data/train_data.mat";
train_tag_path = "data/train_tag.mat";
val_data_path = "data/val_data.mat";
val_tag_path = "data/val_tag.mat";
test_data_path = "data/test_data.mat";
test_tag_path = "data/test_tag.mat";

train_data = load(train_data_path);
train_data = train_data.train_data;
train_data = reshape(train_data, [28, 28, length(train_data)/batch_size, batch_size]);

train_tag = load(train_tag_path);
train_tag = train_tag.train_tag;
train_tag = reshape(train_tag, [10, length(train_tag)/batch_size, batch_size]);

val_data = load(val_data_path);
val_data = val_data.val_data;

val_tag = load(val_tag_path);
val_tag = val_tag.val_tag;

test_data = load(test_data_path);
test_data = test_data.test_data;

test_tag = load(test_tag_path);
test_tag = test_tag.test_tag;

sz_train = size(train_data);

randn('seed',1111)
W1 = randn(9, 9, 20);
rand('seed',1111)
W3 =  (2 * rand(100, 2000) - 1) / 20;
W4 = (2 * rand(10, 100) - 1) / 10;
all_val_loss = zeros(1, epochs);
all_precision = zeros(1, epochs);
for epoch =  1:epochs
    tic
    for batch = 1:sz_train(3)
        [W1, W3, W4] = train(W1, W3, W4, train_data(:,:,batch,:), train_tag(:,batch,:), lr);
    end
    toc
    [val_loss, precision]= evaluation(W1, W3, W4, val_data, val_tag);

    all_val_loss(epoch) = val_loss;
    all_precision(epoch) = precision;
    fprintf("| End of Epoch %d |  Validation Loss %e | Precision %.5f\n",...
                    epoch, val_loss, precision)
end

[test_loss, precision] = evaluation(W1, W3, W4, test_data, test_tag);
fprintf("| End of training |  Validation Loss %e | Precision %.5f\n",...
                     test_loss, precision)
result(all_val_loss, all_precision, W1) 