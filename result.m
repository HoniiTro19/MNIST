function result(all_val_loss, all_precision, W1)
figure
epochs = length(all_val_loss);
plot(all_val_loss)
figure
plot(all_precision)
figure
for i = 1:length(W1)
    subplot(4, 5, i)
    imshow(W1(:,:,i),'InitialMagnification','fit')
end
