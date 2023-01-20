function z = paramstester(x, y, ends)

addpath('sc-toolbox')

p = polygon(x, y);
m = stripmap(p, ends);
z = m.prevertex;

end

