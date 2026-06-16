%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%needs two variables data_matrix_splitplot and data_matrix_anova Make sure
%you have equal number of trials in all conditions data_matrix_splitplot =
%sortrows(data_matrix,1); data_matrix_anova = data_matrix_splitplot(:,2:3);
%
% IS:09.02.2011 :: Updated - I/O
%
% the input for that function should be a matrix: -> each row is a trial ->
% first column is the condition code [1 : NumberOfConditionsYouHave] ->
% second column is the firing rate during the baseline period -> third
% column is the firing rate during the reponse period
%
% it returns a structure in which the most important are 3 the values
% structStat.PB (the main effect of the factor RESPONSE: response vs.
% baseline) and structStat.PAB (the significance of the interaction between
% the factor RESPONSE and CONDITION). If at least one of these two values
% is smaller than the statistical threshold (e.g. 0.05), you declare that
% particular cell as having significant response.
%
% * Have in mind that cell which have an inhibitory response (reponse <
% baseline) will bi significant too. You should test additionally whether
% the response is excitatory or inhibitory. Ask Rufin about that.


function structStat = splitplotANOVA(data_matrix_splitplot)

Data = data_matrix_splitplot(1:size(data_matrix_splitplot,1), 2:size(data_matrix_splitplot,2));
Groups = data_matrix_splitplot(1:size(data_matrix_splitplot,1), 1:1);
nwg = size(Data,2);
% nbg = max(Groups);
gps = unique(Groups);
nbg = numel(unique(Groups));
gIndices = 1:nbg;
for i1 =1:nbg
    Groups(gps(i1)==Groups) = gIndices(i1);
end
n = size(Data,1)/nbg;
Datasq = Data.^2;
ABS = sum(sum(Datasq));
som1 = sum(Data);
X = ((sum(som1)).^2)/(nwg*size(data_matrix_splitplot,1));
som2 = sum(Data,2);
AS = sum((som2.^2)/nwg);
levela = zeros(nbg,1);
for i = 1:size(data_matrix_splitplot,1)
    levela(Groups(i,1),1) = levela(Groups(i,1),1) + som2(i);
end;
A = sum((levela.^2)/(n*nwg));
B = sum((som1.^2)./size(Data,1));
combcond = zeros(nbg,nwg);
for i = 1:size(Groups,1)
    for j = 1:nwg
        combcond(Groups(i,1), j) = combcond(Groups(i,1),j) + Data(i,j);
    end;
end;
AB = sum(sum((combcond.^2)/n));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sum of Squares
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SS_totaal = ABS - X;
SS_betweensubj = AS - X;
SS_A = A - X;
SS_SwA = AS - A;
SS_withinsubj = ABS - AS;
SS_B = B - X;
SS_AB = AB - A - B + X;
SS_BSwA = ABS - AB - AS + A;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
df_totaal = (n*nbg*nwg)-1;
df_betweensubj = (n*nbg)-1;
df_A = nbg-1;
df_SwA = nbg*(n-1);
df_withinsubj = n*nbg*(nwg-1);
df_B = nwg - 1;
df_AB = (nbg-1)*(nwg-1);
df_BSwA = nbg*(n-1)*(nwg-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MS_A = SS_A/df_A;
MS_SwA = SS_SwA/df_SwA;
MS_B = SS_B/df_B;
MS_AB = SS_AB/df_AB;
MS_BSwA = SS_BSwA/df_BSwA;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FA = MS_A/MS_SwA;
FB = MS_B/MS_BSwA;
FAB = MS_AB/MS_BSwA;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PA = 1-fcdf(FA,df_A,df_SwA);
PB = 1-fcdf(FB,df_B,df_BSwA);
PAB = 1-fcdf(FAB,df_AB,df_BSwA);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output :: IS:09.02.2011
structStat.PA = PA;
structStat.PB = PB;
structStat.PAB = PAB;
structStat.FA = FA;
structStat.FB = FB;
structStat.FAB = FAB;
structStat.SS_A = SS_A;
structStat.SS_B = SS_B;
structStat.SS_AB = SS_AB;
structStat.MS_A = MS_A;
structStat.MS_B = MS_B;
structStat.MS_AB = MS_AB;
structStat.df_A = df_A;
structStat.df_B = df_B;
structStat.df_AB = df_AB;
structStat.df_SwA = df_SwA;
structStat.df_BSwA = df_BSwA;
structStat.df_betweensubj = df_betweensubj;
structStat.df_totaal = df_totaal;
structStat.df_withinsubj = df_withinsubj;
if PB < 0.05 || PAB < 0.05
    structStat.significant = 1;
else
    structStat.significant = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Printing the p values to respective files

% if size(trials,2)==16
%     fprintf(        '%s\t%0.3f\t%0.3f\t%0.3f\t', fnme,PA,PB,PAB);
%     fprintf(    '\n');
% elseif size(trials,2)==5
%     fprintf(        '%s\t%0.3f\t%0.3f\t%0.3f\t', fnme,PA,PB,PAB);
%     fprintf(    '\n');
% end

% end of DoSplitplotANOVA(..) fn





% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% trials(1,max(dat(:,1))) = 0; for i = 1:size(dat,1)
%     for j = 1:size(trials,2)
%         if dat(i,1)==j
%             trials(1,j) = trials(1,j)+1;
%         end
%     end
% end
%
% conds{1,size(trials,2)} = []; for i = 1:size(dat,1)
%     for j = 1:size(trials,2)
%         if dat(i,1)==j
%             trialdata = dat(i,:); conds{1,j} = [conds{1,j};trialdata];
%             clear trialdata
%         end
%     end
% end data_matrix = []; for i = 1:size(conds,2)
%     data_matrix = [data_matrix;conds{i}(1:min(trials(1,:)),:)];
% end data_matrix = [data_matrix(:,1) data_matrix(:,5:6)];