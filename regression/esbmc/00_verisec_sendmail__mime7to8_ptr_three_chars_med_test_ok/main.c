#include "../stubs.h"
#include "../base.h"

#define MAXLINE BASE_SZ

int main (void)
{
  char fbuf[MAXLINE+1];
  char *fbufp;
  int c1, c2, c3;

  fbufp = fbuf;

  while ((c1 = nondet_int ()) != EOF)
  {
    c2 = nondet_int ();
    if (c2 == EOF)
      break;

    c3 = nondet_int ();
    if (c3 == EOF)
      break;

    if (c1 == '=' || c2 == '=')
      continue;

    /* OK */
    *fbufp = c1;

    /* OK */
    if (*fbufp++ == '\n' || fbufp >= &fbuf[MAXLINE])
    {
      if (fbufp <= fbuf)
	fbufp = fbuf;
      else if (*--fbufp != '\r') 
	fbufp++;

      /* OK */
      *fbufp = EOS;
      fbufp = fbuf;
    }

    /* OK */
    *fbufp = c2;

    /* OK */
    if (*fbufp++ == '\n' || fbufp >= &fbuf[MAXLINE])
    {
      if (fbufp <= fbuf)
	fbufp = fbuf;
      else if (*--fbufp != '\r') 
	fbufp++;

      /* OK */
      *fbufp = EOS;
      fbufp = fbuf;
    }

    if (c3 == '=')
      continue;
    /* OK */
    *fbufp = c3;

    /* OK */
    if (*fbufp++ == '\n' || fbufp >= &fbuf[MAXLINE])
    {
      if (fbufp <= fbuf)
	fbufp = fbuf;
      else if (*--fbufp != '\r') 
	fbufp++;

      /* OK */
      *fbufp = EOS;
      fbufp = fbuf;
    }
  }

  /* force out partial last line */
  if (fbufp > fbuf)
  {
    /* OK */
    *fbufp = EOS;
  }

  return 0;
}
