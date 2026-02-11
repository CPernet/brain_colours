function saveCMapSub(pth, prefix, fnm, lut)

% Chris Rorden

[~,n] = fileparts(fnm);
fid = fopen(fullfile(pth,[prefix, n,'.cmap']),'w');
fprintf(fid,'%0.6f %0.6f %0.6f\n',lut');
fclose(fid);
%end saveCMapSub()

