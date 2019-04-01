function [normalized_a] = DEFARTMAP_Normalize(a)

% added: 062012-Alex; normalization function not available in original BARTMAPcode folder
% OGI MODIFICATION
% Go from 0 to 1, for ART
%  
% VERTICAL is N the number of data points
% Horizontal is M, the features
% NOTE THIS IS the OPPOSITE of complement code, but I have already 
% coded so much, that all my legacy code would have to be changed...

if nargin > 1
  error('Wrong number of arguments.');
end

minp = min(a')';
maxp = max(a')';
[R,Q]=size(a);
oneQ = ones(1,Q);

equal = minp==maxp;
%baddata = find(equal)
nequal = ~equal;
if sum(equal) ~= 0
  minp0 = minp.*nequal - 1*equal;
  maxp0 = maxp.*nequal + 1*equal;
else
  minp0 = minp;
  maxp0 = maxp;
end

normalized_a = (a-minp0*oneQ)./((maxp0-minp0)*oneQ);
if sum(equal) ~= 0
    eq=find(equal);
    normalized_a(eq,:)=0;
end