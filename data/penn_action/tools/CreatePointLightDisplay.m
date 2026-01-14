% CREATEPOINTLIGHTDISPLAY
% DESCRIPTION: Create point light display video volume based on input annotation
% PARAMETERS:
%    - annotation: annotation entry
%    - radius: radius of joint display points
%    - width: line width of joint display marker
% USAGE:  h = figure; 
%         vol = CreatePointLightDisplay(annotations,5,3);
% 
% CODED BY: Kosta Derpanis
% EMAIL: kosta.derpanis@gmail.com
% DATE: May 30, 2012
%
function vol = CreatePointLightDisplay(annotation, radius, width)

   if (nargin < 2)
       radius = 1;
       width  = 3;
   end
  
   dims = annotation.dimensions;
   vol = zeros(dims);
    
   X = round(annotation.x);
   Y = round(annotation.y);
   [tmp T] = meshgrid(1:size(X,2), 1:size(X,1));
   
   X = X(:);
   Y = Y(:);
   T = T(:);
   visibility = annotation.visibility(:);
   
   ind = find(visibility == 0);
   
   X(ind) = [];
   Y(ind) = [];
   T(ind) = [];
   
   for w = -width:width
       for i = 0:radius-1 
          ind = sub2ind(dims, Y+w, min( [X+i repmat(dims(2), numel(X), 1)], [], 2), T);  
          vol(ind) = 255;

          ind = sub2ind(dims, Y+w, max( [X-i ones(numel(X), 1)], [], 2), T);  
          vol(ind) = 255; 

          ind = sub2ind(dims, min( [Y+i repmat(dims(1), numel(Y), 1)], [], 2) , X+w, T);  
          vol(ind) = 255;

          ind = sub2ind(dims, max( [Y-i ones(numel(Y), 1)], [] ,2), X+w, T);  
          vol(ind) = 255;
       end
   end
