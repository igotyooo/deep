function [ rect ] = mammo_cutter( img )
margin = 50;
I = img(30:end-30,30:end-30,:);
BW = im2bw(I, graythresh(I));
se = strel('disk',17);
BW = imdilate(BW,se);
[B,~] = bwboundaries(BW,'noholes');
s  = regionprops(BW, 'Area');
for k = 1:length(B)
    S(k)=s(k).Area;
end
[~,in_max]=max(S);
boundary = B{in_max};

left = min( boundary(:,2) ) - margin - 30;
right = max( boundary(:,2) ) + margin + 30;
top  = min( boundary(:,1) ) - margin - 30;
bottom  = max( boundary(:,1) ) + margin + 30;

if left < 1
    left = 1;
end
if top < 1
    top = 1;
end
if right > size(img,2)
    right = size(img,2);
end
if bottom > size(img,1)
    bottom = size(img,1);
end
rect = [ left top right-left+1 bottom-top+1 ];

end

