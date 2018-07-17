% This is the matlab script used to 'fix' the luminance issue in colour
% maps used in human brain tomographic imaging
%
% - initial colour maps taken from MRICro (https://www.nitrc.org/projects/mricron)
% - .lut fules are read using Matthew Brett's pr_getcmap.m function (slover)
% - the luminance is fixed using equalisecolourmap.m provided by Peter Kovesi
% (Peter Kovesi. Good Colour Maps: How to Design Them. arXiv:1509.03700)
% https://www.peterkovesi.com/matlabfns/index.html#colour
% also depends on http://www2.ece.rochester.edu/~gsharma/ciede2000/
% - the vizualization is done using are RGB to LAB space converion using the
% colorspace.m and colormapline.m function from Matteo Niccoli
% (https://mycarta.wordpress.com/2012/05/12/the-rainbow-is-dead-long-live-the-rainbow-part-1/)
%
% Cyril Pernet 16 July 2018

current = pwd;
%% scan for colour map taken from mricro ()

cd('D:\MRI\mricron\lut')
files = dir('*.lut');

%% now read, correct luminance, and save
cd(current);
mkdir('new_braincolour_maps')

for map = 1:size(files,1)
    lutmap = loadLutSub([files(map).folder filesep files(map).name]);
    try
        lutmap2 = equalisecolourmap('RGB', lutmap, 'CIE76', [1 0 0], 1/25*length(lutmap), 0,0);
        csvwrite(['new_braincolour_maps' filesep files(map).name(1:end-4) '.csv'],lutmap2)
        save(['new_braincolour_maps' filesep files(map).name(1:end-4)],'lutmap2')
        saveLutSub('new_braincolour_maps', [], files(map).name(1:end-4), lutmap2)
        saveCMapSub('new_braincolour_maps', [], files(map).name(1:end-4), lutmap2)

        lutmap2 = equalisecolourmap('RGB', lutmap, 'CIE76', [1 1 1], 1/25*length(lutmap), 0,0);
        csvwrite(['new_braincolour_maps' filesep files(map).name(1:end-4) '_iso.csv'],lutmap2)
        save(['new_braincolour_maps' filesep files(map).name(1:end-4) '_iso'],'lutmap2')
        saveLutSub('new_braincolour_maps', [], [files(map).name(1:end-4) '_iso'], lutmap2)
        saveCMapSub('new_braincolour_maps', [], [files(map).name(1:end-4) '_iso'], lutmap2)
    end
end

%% add two diverging colour maps
[dmap, name, desc] = cmap('D1');
csvwrite(['new_braincolour_maps' filesep 'diverging_bwr.csv'],dmap)
save(['new_braincolour_maps' filesep 'diverging_bwr'],'dmap')
saveLutSub('new_braincolour_maps', [],'diverging_bwr', dmap)
saveCMapSub('new_braincolour_maps', [], 'diverging_bwr', dmap)

        
[dmap, name, desc] = cmap('D7');
csvwrite(['new_braincolour_maps' filesep 'diverging_bgy.csv'],dmap)
save(['new_braincolour_maps' filesep 'diverging_bgy'],'dmap')
saveLutSub('new_braincolour_maps', [],'diverging_bgy', dmap)
saveCMapSub('new_braincolour_maps', [], 'diverging_bgy', dmap)

%% make a figure to illustrate
figure; index =1;
for map = [1  10 12 13]
    lutmap = loadLutSub([files(map).folder filesep files(map).name]);
    cd(current)
    lutmap2 = equalisecolourmap('RGB', lutmap, 'CIE76', [1 0 0], 1/25*length(lutmap), 0,0);
    subplot(2,4,index)
    LS=colorspace('RGB->Lab',lutmap);
    h=colormapline(1:1:length(LS),LS(:,1),[],lutmap);
    set(h,'linewidth',2); grid on; box on; axis square;
    ylabel('Luminance')
    title ([files(map).name(1:end-4)],'Color','k','FontSize',12);
    subplot(2,4,index+4);
    LS=colorspace('RGB->Lab',lutmap2);
    h=colormapline(1:1:length(LS),LS(:,1),[],lutmap2);
    set(h,'linewidth',2); grid on; box on; axis square;
    xlabel('Colour level'); ylabel('Luninance')
    index = index+1;
end
