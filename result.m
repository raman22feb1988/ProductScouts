function[result] = result()
load('data.mat');
z = cell2mat(x(:, 2));
result = [(0:max(z))' zeros(max(z) + 1, 1)];
for i = 1:size(z, 1)
if(y(i, 1) > result(z(i, 1) + 1, 2))
result(z(i, 1) + 1, 2) = y(i, 1);
end
% disp(i);
save('result.mat', 'result');
end
end