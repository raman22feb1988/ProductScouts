function[out] = classifier(threshold)
load('data.mat');
out = zeros(size(in, 1), 1);
z = cell2mat(in(:, 2));
w = cell2mat(x(:, 2));
v = [x num2cell(y)];
result = [(0:max(w))' zeros(max(w) + 1, 1)];
for i = 1:size(w, 1)
if(y(i, 1) > result(w(i, 1) + 1, 2))
result(w(i, 1) + 1, 2) = y(i, 1);
end
% disp(i);
end
for i = 1:size(in, 1)
r = v(w == z(i, 1), 1:3);
if size(r, 1) > 0
if size(r, 1) <= threshold
q = in(i, 1);
p = r(:, 1);
[m, n] = strnearest(q, p);
o = cell2mat(m);
s = o(1);
t = cell2mat(r(s, 3));
% out(i, 1) = result(result(:, 1) == z(i, 1), 2);
else
p = r(:, 3);
s = cell2mat(p);
t = max(s);
end
out(i, 1) = t;
end
% disp(i);
end
% save('classifier.mat', 'out');
end