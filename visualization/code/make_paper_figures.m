function make_paper_figures
% Publication figures from saved, verified results only. MATLAB native graphics.
root=fileparts(fileparts(mfilename('fullpath'))); dd=fullfile(root,'data');
out=fullfile(root,'figures');
set(groot,'defaultAxesFontName','Times New Roman','defaultTextFontName','Times New Roman', ...
    'defaultAxesFontSize',10,'defaultTextFontSize',10,'defaultAxesLineWidth',.7, ...
    'defaultAxesTickDir','out','defaultAxesBox','off','defaultFigureColor','w', ...
    'defaultTextInterpreter','tex','defaultLegendInterpreter','none', ...
    'defaultAxesColor','w','defaultAxesXColor',[.12 .16 .20], ...
    'defaultAxesYColor',[.12 .16 .20],'defaultAxesZColor',[.12 .16 .20], ...
    'defaultTextColor',[.12 .16 .20],'defaultLegendTextColor',[.12 .16 .20]);
C=[.23 .43 .56;.50 .40 .66;.84 .34 .34;.22 .60 .64];
S=readtable(fullfile(dd,'policy_test_summary.csv'),'TextType','string');
names=["no_update","half_gain","full_gain","no_reserve"];
S=S(arrayfun(@(n)find(S.policy==n),names),:);
labels={'No update','Half gain','Selected','No reserve'};

f=canvas(18.2,7.4); t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t); a=[S.adjustment_cost_yuan S.emergency_cost_yuan]/1e3;
b=bar(ax,1:4,a,'stacked','BarWidth',.65);b(1).FaceColor=C(1,:);b(2).FaceColor=C(3,:);
set(ax,'XTick',1:4,'XTickLabel',labels,'XTickLabelRotation',18);
ylabel(ax,'Controllable cost (10^3 CNY)');ylim(ax,[0 280]);
title(ax,'(a) Settlement decomposition','FontWeight','normal');gridy(ax);
legend(ax,{'Adjustment','Emergency'},'Location','northoutside','Orientation','horizontal','Box','off','FontSize',10);
text(ax,.5,252,'Common plan charge: 4.1581 million CNY','FontSize',10);
ax=nexttile(t);b=bar(ax,1:4,S.emergency_kwh/1e3,.65,'FaceColor','flat');b.CData=C;
set(ax,'XTick',1:4,'XTickLabel',labels,'XTickLabelRotation',18);
ylabel(ax,'Emergency energy (MWh)');ylim(ax,[0 45]);gridy(ax);
title(ax,'(b) Emergency procurement','FontWeight','normal');
for i=1:4,text(ax,i,S.emergency_kwh(i)/1e3+2,sprintf('%d days',S.emergency_days(i)),'HorizontalAlignment','center','FontSize',10);end
savefigs(f,out,'comparison');

F=readtable(fullfile(dd,'forecast_comparison.csv'),'TextType','string');
F=F(F.split=="test",:);
V=readtable(fullfile(dd,'policy_validation_summary.csv'),'TextType','string');
f=canvas(18.2,7.4);t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t);b=bar(ax,[F.mae_yuan_per_kwh F.rmse_yuan_per_kwh],'grouped');
b(1).FaceColor=C(1,:);b(2).FaceColor=C(2,:);gridy(ax);
set(ax,'XTick',1:4,'XTickLabel',{'Causal','HGB','HGB-28','GRU'},'XTickLabelRotation',15);
ylabel(ax,'Price error (CNY/kWh)');ylim(ax,[0 .155]);
title(ax,'(a) Test forecasting error','FontWeight','normal');
legend(ax,{'MAE','RMSE'},'Location','northoutside','Orientation','horizontal','Box','off');
ax=nexttile(t);hold(ax,'on'); base=V.total_cost_yuan(V.policy=="full_gain");
for j=1:3
    prefixes=["hgb","hgb_history","gru"]; y=zeros(1,4);
    alphas=[.25 .5 1];
    for i=2:4, key=prefixes(j)+"_alpha_"+string(alphas(i-1));y(i)=V.total_cost_yuan(V.policy==key)-base;end
    plot(ax,[0 .25 .5 1],y,'-o','Color',C(j,:),'LineWidth',1.4,'MarkerSize',4,'MarkerFaceColor','w');
end
gridy(ax);xlim(ax,[-.03 1.03]);xticks(ax,[0 .25 .5 1]);ylim(ax,[-120 2350]);
xlabel(ax,'Learned-price fusion weight');ylabel(ax,'Validation cost excess (CNY)');
title(ax,'(b) Decision-based selection','FontWeight','normal');
legend(ax,{'HGB','HGB-28','GRU'},'Location','northoutside','Orientation','horizontal','Box','off','FontSize',10);
savefigs(f,out,'forecast_decision');

D=readtable(fullfile(dd,'paired_daily.csv'),'TextType','string');
f=canvas(18.2,7.4);t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t); x=(1:height(D))';
area(ax,x,D.cumulative_saving_yuan/1e3,'FaceColor',C(1,:),'FaceAlpha',.18,'EdgeColor',C(1,:),'LineWidth',1.4);hold(ax,'on');yline(ax,0,':','Color',[.5 .5 .5]);
gridy(ax);ylabel(ax,'Cumulative saving (10^3 CNY)');xlim(ax,[x(1) x(end)]);
xticks(ax,[1 32 62 92]);xticklabels(ax,{'Oct 1','Nov 1','Dec 1','Dec 31'});
title(ax,'(a) Paired chronological savings','FontWeight','normal');
ax=nexttile(t);vals=sort(D.daily_saving_yuan/1e3);n=numel(vals);
stairs(ax,vals,(1:n)/n,'Color',C(2,:),'LineWidth',1.5);hold(ax,'on');xline(ax,0,':','Color',[.4 .4 .4]);gridy(ax);
xlabel(ax,'Daily saving (10^3 CNY)');ylabel(ax,'Empirical cumulative probability');ylim(ax,[0 1.03]);
title(ax,'(b) Heterogeneous daily effects','FontWeight','normal');
text(ax,.97,.28,sprintf('23 cheaper | 24 dearer | 45 ties\nMean saving: 260.91 CNY/day\n7-day block 95%% interval:\n[38.92, 576.44] CNY/day'),'Units','normalized','HorizontalAlignment','right','FontSize',10);
savefigs(f,out,'paired_effect');

ST=readtable(fullfile(dd,'stress_summary.csv'),'TextType','string');
cases=["unshocked","load_plus_10pct","pv_minus_20pct","price_plus_20pct","joint_shock"];
A=zeros(5,2);E=A;days=A;
for i=1:5
    for j=1:2
        gains=[.5 1];p=ST(ST.split==cases(i)&ST.gain==gains(j),:);
        A(i,j)=p.total_cost_yuan/1e6;E(i,j)=p.emergency_kwh/1e3;days(i,j)=p.emergency_days;
    end
end
f=canvas(18.2,7.4);t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t);b=bar(ax,A,'grouped');b(1).FaceColor=[.65 .70 .75];b(2).FaceColor=C(1,:);
set(ax,'XTick',1:5,'XTickLabel',{'Base','Load +10%','PV -20%','Price +20%','Joint'},'XTickLabelRotation',20);
ylabel(ax,'Total cost (million CNY)');ylim(ax,[0 15]);gridy(ax);
title(ax,'(a) Fixed-policy stress replay','FontWeight','normal');
legend(ax,{'Half gain','Selected'},'Location','northoutside','Orientation','horizontal','Box','off');
ax=nexttile(t);b=bar(ax,E,'grouped');b(1).FaceColor=[.77 .66 .78];b(2).FaceColor=C(3,:);
set(ax,'XTick',1:5,'XTickLabel',{'Base','Load +10%','PV -20%','Price +20%','Joint'},'XTickLabelRotation',20);
ylabel(ax,'Emergency energy (MWh)');gridy(ax);ylim(ax,[0 1120]);
title(ax,'(b) Emergency exposure under shocks','FontWeight','normal');
legend(ax,{'Half gain','Selected'},'Location','northoutside','Orientation','horizontal','Box','off');
savefigs(f,out,'stress');

% Daily response matrix: one cell per actual test day, no invented interpolation.
T=readtable(fullfile(dd,'policy_test_daily.csv'),'TextType','string');
half=T(T.split=="half_gain",:);full=T(T.split=="full_gain",:);none=T(T.split=="no_update",:);nr=T(T.split=="no_reserve",:);
M=[(half.total_cost_yuan-none.total_cost_yuan)';(half.total_cost_yuan-full.total_cost_yuan)';(half.total_cost_yuan-nr.total_cost_yuan)']/1e3;
f=canvas(18.2,7.4);t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t);imagesc(ax,1:92,1:3,M);set(ax,'YTick',1:3,'YTickLabel',{'No update','Selected','No reserve'},'XTick',[1 32 62 92],'XTickLabel',{'Oct 1','Nov 1','Dec 1','Dec 31'});
cmap=interp1([0 .5 1],[C(3,:);1 1 1;C(1,:)],linspace(0,1,256));colormap(ax,cmap);clim(ax,[-8 8]);
clim(ax,[-ceil(max(abs(M(:)))) ceil(max(abs(M(:))))]);
cb=colorbar(ax);cb.Label.String='Saving vs half gain (10^3 CNY)';cb.FontSize=10;cb.Color=[.12 .16 .20];
title(ax,'(a) Paired daily response matrix','FontWeight','normal');xlabel(ax,'Chronological test date');
ax=nexttile(t);hold(ax,'on');
costchange=(nr.total_cost_yuan-full.total_cost_yuan);emchange=(nr.emergency_kwh-full.emergency_kwh);
scatter(ax,emchange,costchange,20,C(4,:),'filled','MarkerFaceAlpha',.65);xline(ax,0,':');yline(ax,0,':');gridy(ax);
xlabel(ax,'Emergency change without reserve (kWh)');ylabel(ax,'Cost change without reserve (CNY)');
title(ax,'(b) Reserve contribution by day','FontWeight','normal');
savefigs(f,out,'ablation');
fprintf('Five MATLAB figure pairs exported: PDF vector and 900-dpi PNG.\n');
revise_selected_figures;
end

function f=canvas(w,h)
f=figure('Visible','off','Units','centimeters','Position',[1 1 w h],'Color','w');
if isprop(f,'Theme'),f.Theme='light';end
end
function gridy(ax)
set(ax,'YGrid','on','XGrid','off','GridAlpha',.12,'Layer','top','TickLength',[.015 .015]);
end
function savefigs(f,out,name)
drawnow;
exportgraphics(f,fullfile(out,[name '.pdf']),'ContentType','vector','BackgroundColor','white');
exportgraphics(f,fullfile(out,[name '.png']),'Resolution',900,'BackgroundColor','white');
savefig(f,fullfile(out,[name '.fig']));close(f);
end
