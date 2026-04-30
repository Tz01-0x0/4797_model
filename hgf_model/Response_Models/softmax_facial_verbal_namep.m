function pstruct = softmax_facial_verbal_namep(pvec)
% Names observation parameters for softmax_facial_verbal.

pstruct = struct;
pstruct.ze1  = pvec(1);
pstruct.beta = pvec(2);

return;
