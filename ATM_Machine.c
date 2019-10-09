#include<stdio.h>
#include<conio.h>
float available_amount=25000000;
int change(int);
float balance(float);
float withdraw(float);
void main()
{
 char ch;
 int i,pin,n,j,k,p,a;
 pin=1111;
 float amt;
 system("cls");
 while(1)
 {
  if(i==10||n==4)
    break;
  else
  {
   for(j=1;j<=3;j++)
   {
      k=pin;
      a=0;
  printf("Please Enter your Pin : ");
  for(i=1;i<=4;i++)
  {
   ch=getch();
   printf("*");
   ch=ch-48;
   a=a*10+ch;
  }
  if(k==a)
  {
   printf("Correct");
   system("cls");
   printf("\t\t\t\tEnter your choice");
   printf("\n1).Balance Enquiry\n2).Cash Withdrawl\n3).Change Pin\n4).Exit");
   printf("\n\t\t\t\tInput your choice : ");
   scanf("%d",&n);
   system("cls");
   switch(n)
   {
    case 1:
    {
     amt=balance(available_amount);
     printf("The available amount is %.2f",amt);
    }
    break;
    case 2:
    {
     amt=withdraw(amt);
     if(amt>=0)
     printf("The remaining balance is %.2f",amt);
     else
     printf("Insufficient Amount Enntered");
    }
    break;
    case 3:
    {
     pin=change(pin);
     printf("\nSuccessfully changed\n");
    }
    break;
    default:
        printf("Thank you for working with us");
        break;
   }
  }
else
    printf("The pin is wrong");
    printf("\n");
   }
  }
 }
}
float balance(float amount)
{
 float amt;
 amt=25000000;
 return(amt);
}
float withdraw(float amt)
{
  printf("Enter amount to be withdrawn : ");
  scanf("%f",&amt);
  system("cls");
  if(amt<=25000000)
  {
   available_amount=available_amount-amt;
   printf("The amount withdrawn is %.2f\n",amt);
   return(available_amount);
  }
  else
    return(-1);
}
int change(int pin)
{
 char ch;
 int i,j,k,a;
 char c,p;
 k=pin;
 printf("your current pin : ");
 for(j=1;j<=3;j++)
 {
  a=0;
  for(i=1;i<=4;i++)
  {
    ch=getch();
    printf("*");
    ch=ch-48;
    a=a*10+ch;
  }
  if(a==k)
  {
    printf("\n\t\tDear user please enter your new pin : ");
    printf("\n\t\t\t\tnew pin : ");
    for(i=1;i<=4;i++)
    {
     c=getch();
     printf("*");
     ch=ch-48;
     a=a*10+ch;
    }
    printf("\n\t\t\t\tconfirm pin : ");
    for(i=1;i<=4;i++)
    {
     p=getch();
     printf("*");
     ch=ch-48;
     a=a*10+ch;
     }
    if(p==c)
        {
         return(p);
        }
    else
        printf("The two entries are not same");
  }
  else
    printf("Wrong pin");
 }
}
