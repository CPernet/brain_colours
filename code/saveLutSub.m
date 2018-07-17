function saveLutSub(pth, prefix, fnm, lut)

% Chris Rorden

[~,n] = fileparts(fnm);
fid = fopen(fullfile(pth,[prefix, n,'.lut']),'wb');
fwrite(fid,lut*255,'uchar');
fclose(fid);
%end saveLutSub()