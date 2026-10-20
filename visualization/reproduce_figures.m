function reproduce_figures(outputFolder)
% Rebuild publication figures in a disposable output directory.
here=fileparts(mfilename('fullpath'));
repo=fileparts(here);
if nargin<1, outputFolder=fullfile(repo,'runs','matlab'); end
if strcmp(outputFolder,here) || startsWith(string(outputFolder),string(fullfile(repo,'results')))
    error('Choose a generated-output directory, not the reference archive.');
end
if ~exist(outputFolder,'dir'), mkdir(outputFolder); end
figureFolder=fullfile(outputFolder,'figures');
if ~exist(figureFolder,'dir'), mkdir(figureFolder); end
codeFolder=fullfile(outputFolder,'code');
if ~exist(codeFolder,'dir'), mkdir(codeFolder); end
copyfile(fullfile(here,'code','*.m'),codeFolder);
copyfile(fullfile(here,'data'),fullfile(outputFolder,'data'));
addpath(codeFolder,'-begin');
cleanup=onCleanup(@()rmpath(codeFolder)); %#ok<NASGU>
make_paper_figures;
revise_selected_figures;
fprintf('Regenerated figures: %s\n',fullfile(outputFolder,'figures'));
end
