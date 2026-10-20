function revise_selected_figures
root=fileparts(fileparts(mfilename('fullpath')));dd=fullfile(root,'data');out=fullfile(root,'figures');
set(groot,'defaultAxesFontName','Times New Roman','defaultTextFontName','Times New Roman', ...
    'defaultAxesFontSize',10,'defaultTextFontSize',10,'defaultAxesLineWidth',.7, ...
    'defaultAxesTickDir','out','defaultAxesBox','off','defaultFigureColor','w', ...
    'defaultTextInterpreter','tex','defaultLegendInterpreter','none', ...
    'defaultAxesColor','w','defaultAxesXColor',[.12 .16 .20], ...
    'defaultAxesYColor',[.12 .16 .20],'defaultTextColor',[.12 .16 .20], ...
    'defaultLegendTextColor',[.12 .16 .20]);
C=[.23 .43 .56;.50 .40 .66;.84 .34 .34;.22 .60 .64];
T=readtable(fullfile(dd,'policy_test_daily.csv'),'TextType','string');
names=["no_update","half_gain","full_gain","no_reserve"];
parts=cell(1,4);
for i=1:4,parts{i}=sortrows(T(T.split==names(i),:),'date');end
half=parts{2};full=parts{3};none=parts{1};nr=parts{4};
M=[(half.total_cost_yuan-none.total_cost_yuan)';(half.total_cost_yuan-full.total_cost_yuan)';(half.total_cost_yuan-nr.total_cost_yuan)']/1e3;
days=datetime(half.date,'InputFormat','yyyy-MM-dd');
firstMonday=datetime(2025,9,29);offset=days-firstMonday;
week=floor(days2num(offset)/7)+1;day=mod(days2num(offset),7)+1;
f=canvas(18.2,14.8);
cmap=interp1([0 .5 1],[C(3,:);1 1 1;C(1,:)],linspace(0,1,512));
lims=ceil(max(abs(M(:))));labels={'No update','Selected full gain','No reserve'};bottom=[.715 .445 .175];
axs=gobjects(1,3);
annotation(f,'textbox',[.06 .944 .44 .045],'String','(a) Paired daily cost calendars','EdgeColor','none','FontName','Times New Roman','FontSize',11,'HorizontalAlignment','center');
for j=1:3
    Z=nan(7,14);
    for k=1:92,Z(day(k),week(k))=M(j,k);end
    assert(sum(isfinite(Z(:)))==92);
    ax=axes(f,'Position',[.09 bottom(j) .34 .209]);axs(j)=ax;
    imagesc(ax,1:14,1:7,Z,'AlphaData',isfinite(Z));axis(ax,'image');
    ax.Color=[.94 .94 .94];
    xlim(ax,[.5 14.5]);ylim(ax,[.5 7.5]);
    set(ax,'YDir','reverse','YTick',1:7,'YTickLabel',{'Mon','Tue','Wed','Thu','Fri','Sat','Sun'}, ...
        'XTick',[1 6 10 14],'XTickLabel',{'Sep 29','Nov 3','Dec 1','Dec 29'},'TickLength',[0 0]);
    colormap(ax,cmap);clim(ax,[-lims lims]);hold(ax,'on');
    for g=.5:14.5,xline(ax,g,'-','Color',[.91 .91 .91],'LineWidth',.35);end
    for g=.5:7.5,yline(ax,g,'-','Color',[.91 .91 .91],'LineWidth',.35);end
    title(ax,labels{j},'FontSize',10,'FontWeight','normal');
    assert(all(abs(ax.DataAspectRatio-[1 1 1])<1e-9));
end
cb=colorbar(axs(3),'southoutside');cb.Position=[.09 .065 .34 .017];cb.Label.String='Saving vs half gain (10^3 CNY)';cb.FontSize=10;
for j=1:3,axs(j).Position=[.09 bottom(j) .34 .209];end

ax=axes(f,'Position',[.59 .175 .385 .70]);hold(ax,'on');
cols=[.55 .60 .65;C(1,:);C(3,:);C(4,:)];styles={'--','-.','-',':'};
for j=1:4
    x=sort(parts{j}.emergency_kwh/1e3);[ux,~,idx]=unique(x);counts=accumarray(idx,1);cdf=cumsum(counts)/numel(x);
    stairs(ax,[0;ux],[0;cdf],'LineStyle',styles{j},'Color',cols(j,:),'LineWidth',1.7);
end
ylabel(ax,'Empirical cumulative probability');xlabel(ax,'Daily emergency energy (MWh)');
xlim(ax,[0 6.4]);ylim(ax,[.5 1.02]);yticks(ax,.5:.1:1);gridy(ax);
title(ax,'(b) Emergency-energy distributions','FontSize',11,'FontWeight','normal');
legend(ax,{'No update','Half gain','Selected','No reserve'},'Location','northoutside','NumColumns',2,'Box','off','FontSize',10);
text(ax,.96,.27,sprintf('Emergency-free days / 92\nNo update: 52  |  Half gain: 57\nSelected: 65  |  No reserve: 64'), ...
    'Units','normalized','HorizontalAlignment','right','FontSize',10,'VerticalAlignment','middle');
exportpair(f,out,'ablation');

ST=readtable(fullfile(dd,'stress_summary.csv'),'TextType','string');
cases=["unshocked","load_plus_10pct","pv_minus_20pct","price_plus_20pct","joint_shock"];
A=zeros(5,2);E=A;
for i=1:5
    for j=1:2
        gains=[.5 1];p=ST(ST.split==cases(i)&ST.gain==gains(j),:);
        A(i,j)=p.total_cost_yuan/1e6;E(i,j)=p.emergency_kwh/1e3;
    end
end
f=canvas(8.8,11.2);t=tiledlayout(f,2,1,'TileSpacing','compact','Padding','compact');
ax=nexttile(t);b=bar(ax,A,'grouped');b(1).FaceColor=[.65 .70 .75];b(2).FaceColor=C(1,:);
set(ax,'XTick',1:5,'XTickLabel',{'Base','Load +10%','PV -20%','Price +20%','Joint'},'XTickLabelRotation',23);
ylabel(ax,'Total cost (million CNY)');ylim(ax,[0 15]);gridy(ax);
title(ax,'(a) Fixed-policy stress replay','FontWeight','normal');
legend(ax,{'Half gain','Selected'},'Location','northoutside','Orientation','horizontal','Box','off');
ax=nexttile(t);b=bar(ax,E,'grouped');b(1).FaceColor=[.77 .66 .78];b(2).FaceColor=C(3,:);
set(ax,'XTick',1:5,'XTickLabel',{'Base','Load +10%','PV -20%','Price +20%','Joint'},'XTickLabelRotation',23);
ylabel(ax,'Emergency energy (MWh)');ylim(ax,[0 1120]);gridy(ax);
title(ax,'(b) Emergency exposure under shocks','FontWeight','normal');
legend(ax,{'Half gain','Selected'},'Location','northoutside','Orientation','horizontal','Box','off');
exportpair(f,out,'stress');
fprintf('Fig.5 square calendar cells and distributions; Fig.7 vertical panels exported.\n');
end
function n=days2num(d),n=hours(d)/24;end
function f=canvas(w,h)
f=figure('Visible','off','Units','centimeters','Position',[1 1 w h],'Color','w');
if isprop(f,'Theme'),f.Theme='light';end
end
function gridy(ax),set(ax,'YGrid','on','XGrid','off','GridAlpha',.12,'Layer','top','TickLength',[.015 .015]);end
function exportpair(f,out,name)
drawnow;exportgraphics(f,fullfile(out,[name '.pdf']),'ContentType','vector','BackgroundColor','white');
exportgraphics(f,fullfile(out,[name '.png']),'Resolution',900,'BackgroundColor','white');
savefig(f,fullfile(out,[name '.fig']));close(f);
end
