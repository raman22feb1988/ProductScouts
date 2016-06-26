function[out] = brand_classify()
load('data.mat');
out = zeros(size(in, 1), 1);
z = cell2mat(in(:, 2));
w = cell2mat(x(:, 2));
v = [x num2cell(y)];
for i = 1:size(in, 1)
r = v(w == z(i), 1:3);
if size(r, 1) > 0
q = in(i, 1);
p = r(:, 1);
[m, n] = strnearest(q, p);
o = cell2mat(m);
s = o(1);
t = cell2mat(r(s, 3));
% p = r(:, 3);
% s = cell2mat(p);
% t = max(s);
out(i, 1) = t;
end
disp(i);
end
end