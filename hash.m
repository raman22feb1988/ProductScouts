function[x_hash, in_hash] = hash()
load('data.mat');
z = cell2mat(x(:, 2));
w = cell2mat(in(:, 2));
a = max(z);
b = max(w);
count = [(0:a)' zeros(a + 1, 1)];
result = [(0:b)' zeros(b + 1, 1)];
for i = 1:size(z, 1)
count(z(i, 1) + 1, 2) = count(z(i, 1) + 1, 2) + 1;
% disp(i);
end
for i = 1:size(w, 1)
result(w(i, 1) + 1, 2) = result(w(i, 1) + 1, 2) + 1;
% disp(i);
end
x_hash = cell(a + 1, 1);
in_hash = cell(b + 1, 1);
for i = 1:a+1
x_hash{i, 1} = cell(count(i, 2), 2);
% disp(i);
end
for i = 1:b+1
in_hash{i, 1} = cell(result(i, 2), 1);
% disp(i);
end
progress = [(0:a)' zeros(a + 1, 1)];
process = [(0:b)' zeros(b + 1, 1)];
for i = 1:size(z, 1)
progress(z(i, 1) + 1, 2) = progress(z(i, 1) + 1, 2) + 1;
x_hash{z(i, 1) + 1, 1}{progress(z(i, 1) + 1, 2), 1} = char(x(i, 1));
x_hash{z(i, 1) + 1, 1}{progress(z(i, 1) + 1, 2), 2} = y(i, 1);
% disp(i);
end
for i = 1:size(w, 1)
process(w(i, 1) + 1, 2) = process(w(i, 1) + 1, 2) + 1;
in_hash{w(i, 1) + 1, 1}{process(w(i, 1) + 1, 2), 1} = char(in(i, 1));
% disp(i);
end
% for i = 1:a+1
% x_hash{i, 1} = [x_hash{i, 1}{:, 1} mat2cell(cell2mat(x_hash{i, 1}{:, 2}))];
% % disp(i);
% end
% for i = 1:b+1
% in_hash{i, 1} = [in_hash{i, 1}{:, 1} mat2cell(cell2mat(in_hash{i, 1}{:, 2}))];
% % disp(i);
% end
save('hash.mat', 'x_hash', 'in_hash');
end