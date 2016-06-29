function[out] = classify_brand(A, B, in)
z = cell2mat(in(:, 2));
C = [A(:, 2) B(:, 2)];
sortrows(C, 1);
for i = 1:size(in, 1)
t = C(find(C(:, 1) == z(i, 1)), 2);
if size(t, 1) > 0
out(i, 1) = t;
end
% disp(i);
end
% save('classify_brand.mat', 'out');
end