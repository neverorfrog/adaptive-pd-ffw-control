function [H] = DH_single_transform(DH_table, index)
    row = DH_table(index,:);
    alpha = row(1);
    d = row(2);
    a = row(3);
    theta = row(4);
    H = [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha),  a * cos(theta);
         sin(theta), cos(theta)*cos(alpha),  -cos(theta)*sin(alpha), a * sin(theta);
         0,          sin(alpha),             cos(alpha),             d;
         0,          0,                      0,                      1];

end

