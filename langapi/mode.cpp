/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <string.h>

// Prototypes

#include "mode.h"

const char *extensions_ansi_c  []={ "c", "i", NULL };
const char *extensions_intrep  []={ NULL };
const char *extensions_pvs     []={ "pvs", NULL };
const char *extensions_vhdl    []={ "vhdl", NULL };
const char *extensions_smv     []={ "smv", "flat_smv", NULL };
const char *extensions_csp     []={ "csp", NULL };
const char *extensions_netlist []={ "ntl", NULL };
const char *extensions_conf    []={ "conf", NULL };
const char *extensions_specc   []={ "sc", "si", NULL };
const char *extensions_promela []={ "promela", NULL };
const char *extensions_xml     []={ "xmi", "xml", NULL };
const char *extensions_pascal  []={ "pas", NULL };

#ifdef _WIN32
const char *extensions_cpp     []={ "cpp", "cu", "cc", "ipp", "cxx", NULL };
#else
const char *extensions_cpp     []={ "cpp", "cu", "cc", "ipp", "C", "cxx", NULL };
#endif

const char *extensions_simplify[]={ "ax", NULL };
const char *extensions_bp      []={ "bp", NULL };
const char *extensions_cvc     []={ "cvc", NULL };
const char *extensions_csharp  []={ "cs", NULL };
const char *extensions_smt     []={ "smt", NULL };
const char *extensions_nsf     []={ "nsf", NULL };
const char *extensions_php     []={ "php", NULL };
const char *extensions_mdl     []={ "mdl", NULL };

/*******************************************************************\

Function: get_mode

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int get_mode(const std::string &str)
{
  unsigned i;

  for(i=0; mode_table[i].name!=NULL; i++)
    if(str==mode_table[i].name)
      return i;

  return -1;
}

/*******************************************************************\

Function: get_mode_filename

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int get_mode_filename(const std::string &filename)
{
  const char *ext=strrchr(filename.c_str(), '.');

  if(ext==NULL) return -1;

  std::string extension=ext+1;

  if(extension=="") return -1;

  int mode;
  for(mode=0; mode_table[mode].name!=NULL; mode++)
    for(unsigned i=0;
        mode_table[mode].extensions[i]!=NULL;
        i++)
      if(mode_table[mode].extensions[i]==extension)
        return mode;

  return -1;
}

/*******************************************************************\

Function: new_language

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

languaget *new_language(const char *mode)
{
  return (*mode_table[get_mode(mode)].new_language)();
}

