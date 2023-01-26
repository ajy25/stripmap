function out = evalinvtester(x, y, ends, x_evalinv, y_evalinv)

addpath('sc-toolbox')

p = polygon(x, y);
m = stripmap(p, ends);
z = m.prevertex;
comp = x_evalinv + i * y_evalinv;
out = m.evalinv(comp);
out(:)

end