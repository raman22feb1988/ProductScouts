function[count, result] = count()
load('data.mat');
z = cell2mat(x(:, 2));
w = cell2mat(in(:, 2));
count = [(0:max(z))' zeros(max(z) + 1, 1)];
result = [(0:max(w))' zeros(max(w) + 1, 1)];
for i = 1:size(z, 1)
count(z(i, 1) + 1, 2) = count(z(i, 1) + 1, 2) + 1;
% disp(i);
end
for i = 1:size(w, 1)
result(w(i, 1) + 1, 2) = result(w(i, 1) + 1, 2) + 1;
% disp(i);
end
save('count.mat', 'count', 'result');
end