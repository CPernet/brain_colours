function lut = loadLutSub(fnm)

% load color lookup table in imageJ/MRIcron lut format or FSLeyes cmap format
% Chris Rorden

if ~exist(fnm,'file'), error('Unable to find %s', fnm); end
stats = dir(fnm);
if stats.bytes == 768 %ImageJ/MRIcron format bytes 0..255 RRR..RGGG..GBBB..B
    fid = fopen(fnm,'rb');
    lut = fread(fid);
    fclose(fid);
    lut = reshape(lut,256,3);
    lut = lut / 255; %scale 0..255 -> 0..1
    return;
end
[~,~,x] = fileparts(fnm);
if strcmpi(x,'.lut'), lut = []; fprintf('Unable to read %s\n', fnm); return; end
fid = fopen(fnm,'r');
lut = fscanf(fid,'%f %f %f');
fclose(fid);
lut = reshape(lut,3,numel(lut)/3)';
%loadLutSub()
if (max(lut(:)) > 1.0) ||  (min(lut(:)) < 0), error('RGB should be in range 0..1 %s', fnm); end
if (size(lut,1) == 255)
    lut = [[0 0 0];lut];
end
if (size(lut,1) < 255)
    %interpolate
    lut = [[0 0 0];lut];
    R = lut(:,1);
    G = lut(:,1);
    B = lut(:,1);
    xq = 1: (numel(R)-1)/255 : numel(R);
    R = interp1(R,xq);
    G = interp1(G,xq);
    B = interp1(B,xq);
    lut = [R' G' B'];
end
%loadLutSub()

