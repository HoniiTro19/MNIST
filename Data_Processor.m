function Data_Processor()
path = "data/MNISTData.mat";
data = load(path);
test_data = data.X_Test;
test_tag = data.D_Test;
train_data = data.X_Train;
train_tag = data.D_Train;

val_rate = 0.01;
train_num = length(train_data);
val_num = round(val_rate * train_num);
idx = randperm(train_num);
val_data = train_data(:, :, idx(1:val_num));
val_tag = train_tag(:, idx(1:val_num));
train_data = train_data(:, :, idx(val_num+1:end));
train_tag = train_tag(:, idx(val_num+1:end));

save data/train_data.mat train_data
save data/train_tag.mat train_tag
save data/test_data.mat test_data 
save data/test_tag.mat test_tag 
save data/val_data.mat val_data 
save data/val_tag.mat val_tag 