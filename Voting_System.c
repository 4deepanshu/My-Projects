#include<stdio.h>
#include<conio.h>
int show(int);
void Admin(int,int,int);
void main()
{
 int p,n,n1,i,pin,reg,k=0,party,id[100],a;
 int party1=0;
 int party2=0;
 int party3=0;
 char ch;
 printf("\n1).Admin");
 printf("\n2).Voter");
 printf("\n3).exit");
 printf("\n\t\t\t\tInput the Choice : ");
 p=show(n);
if(p==1)
 {
    Admin(party1,party2,party3);
 }
   else if(p==2)
    {
      while(1)
      {
       printf("Enter registration no. : ");
       scanf("%d",&reg);
       system("cls");
       int found=0;
       for(i=0;i<=k;i++)
       {
        if(reg==id[i])
            found=1;
       }
        if(found==0)
          {
           id[k++]=reg;
           printf("\t\t\t\tChoose the party");
           printf("\n1).BJP");
           printf("\n2).Congress");
           printf("\n3).Aam Aadmi\n");
           scanf("%d",&party);
           system("cls");
           if(party==1)
             party1++;
           else if(party==2)
             party2++;
           else if(party==3)
             party3++;
           else
            break;
         }
    printf("Enter 1 to continue and 0 to break");
    scanf("%d",&n1);
    system("cls");
    if(n1==1)
     continue;
    else
      {
        printf("Enter your choice : ");
        p=show(n);
        if(p==1)
         Admin(party1,party2,party3);
      }
    }
  }
  else
    printf("Please Exit the area");
}
void Admin(int party1,int party2,int party3)
{
  char ch;
  int a=0,i,pin;
  pin=5678;
  printf("Enter your four digit pin : ");
  for(i=1;i<=4;i++)
  {
    ch=getch();
    printf("*");
    ch=ch-48;
    a=(a*10)+ch;
  }
   if(party1==party2&&party2==party3)
    {
     printf("\nThere will be re-election");
    }
    else if(party1!=party2||party2!=party3||party3!=party1)
    {
     if(a==pin)
     {
      if(party1>party2)
      {
        if(party1>party3)
            printf("\nThe winner is BJP\n");
        else
            printf("\nThe winner is AAM AADMI Party\n");
      }
       else if(party2>party3)
            printf("\nThe winner is congress\n");
       else
        printf("\nThe winner is AAM AAdMI Party\n");
     }
    }
     else
        printf("\nThe pin is Wrong");
}
int show(int n)
{
 scanf("%d",&n);
 return(n);
 system("cls");
}


