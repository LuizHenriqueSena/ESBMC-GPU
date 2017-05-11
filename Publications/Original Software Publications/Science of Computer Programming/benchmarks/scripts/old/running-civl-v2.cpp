/*******************************************************************

 Author: Felipe R. Monteiro

 Date: Jan 2017

 Description: Verify ESBMC-GPU test suite with CIVL and get its
              verification time

 \*******************************************************************/

#include <iostream>
#include <string>
#include <fstream>
#include <list>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
using namespace std;

#define BILLION 1E9
#define SUCCESSFUL 1
#define FAILED 2

unsigned short int CORRECT = 0;
unsigned short int INCORRECT = 0;
unsigned short int FALSE_CORRECT = 0;
unsigned short int FALSE_INCORRECT = 0;
unsigned short int NOT_SUPPORTED = 0;

/*
   Function: getDirectoryList()
   Description: Get a list of directory names
*/
list<string> getDirectoryList(){

   string dir;
	ifstream directories;
   list<string> listOfDirectories;

   system("ls >> directories.txt");
   
   directories.open("directories.txt");
   while(!(directories.eof()))
   {
	   getline(directories,dir);
      if(dir != "directories.txt" && dir != "" && dir != "make.sh" && dir != "running-civl.cpp" && dir != "run" && dir != "running-civl.cpp~")
	      listOfDirectories.push_back(dir);
   }
	directories.close();
   system("rm directories.txt");

   return listOfDirectories;
}

/*
   Function: run_CIVL()
   Description: Analyse verification results
*/
void run_CIVL(){
   string res;
   string testDesc;
	ifstream results;
   int tempResults = -1;

   system("civl verify main.cu >> results.txt"); // to run CIVL
   
   results.open("results.txt");
   while(!(results.eof()))
   {
	   getline(results,res);
         
      if(res == "=== Result ==="){
         getline(results,res);
         if(res == "The standard properties hold for all executions."){
            tempResults = SUCCESSFUL;
         } else {
            tempResults = FAILED;
	 }
	 break;
      }
   }
   results.close();
   res.clear();
   system("rm results.txt");

   if (tempResults == FAILED)
      system("rm -r CIVLREP/");

   results.open("test.desc");

   while(!(results.eof()))
   {
      getline(results,res);
         
      if(res.at(0) == '^'){
         testDesc = res.substr(1,res.size()-2);
         break;
      }
   }
   results.close();
   
   if(tempResults == -1){
      NOT_SUPPORTED++; cout << "NOT_SUPPORTED";
   } else if(tempResults == SUCCESSFUL && testDesc == "VERIFICATION SUCCESSFUL"){
      CORRECT++; cout << "CORRECT";
   } else if(tempResults == SUCCESSFUL && testDesc == "VERIFICATION FAILED"){
      FALSE_CORRECT++; cout << "FALSE_CORRECT";
   } else if(tempResults == FAILED && testDesc == "VERIFICATION SUCCESSFUL"){
      FALSE_INCORRECT++; cout << "FALSE_INCORRECT";
   } else if(tempResults == FAILED && testDesc == "VERIFICATION FAILED"){
      INCORRECT++; cout << "INCORRECT";
   }
}

int main(){
   timespec start, end, inner_start, inner_end;
   list<string> suites;
   list<string> subSuites;
   list<string> testCases;

   suites = getDirectoryList();

   clock_gettime(CLOCK_MONOTONIC_RAW, &start); // set total start time

   for(list<string>::iterator it = suites.begin(); it != suites.end(); it++){

 	   chdir((*it).c_str());
      cout << "========== SUITE: " << *it << " ========== " << endl << endl;
      
      subSuites = getDirectoryList();

      for (list<string>::iterator it1 = subSuites.begin(); it1 != subSuites.end(); it1++){
         
         chdir((*it1).c_str());
         cout << "====== SUBSUITE: " << *it1 << " ====== " << endl << endl;
         
         testCases = getDirectoryList();
         
         clock_gettime(CLOCK_MONOTONIC_RAW, &inner_start);

         for (list<string>::iterator it2 = testCases.begin(); it2 != testCases.end(); it2++){
            chdir((*it2).c_str());
            run_CIVL(); cout << " ---- " << *it2 << endl;
            chdir("../.");
         }

         clock_gettime(CLOCK_MONOTONIC_RAW, &inner_end);

         cout << "TIME: " << (inner_end.tv_sec - inner_start.tv_sec) + 
                                        (inner_end.tv_nsec - inner_start.tv_nsec)/BILLION << "s" << endl;
         cout << "=====================================" << endl;

         testCases.clear();
         cout << endl;
         chdir("../.");
      }

      subSuites.clear();
      cout << endl << endl;      
	   chdir("../.");
   }

   clock_gettime(CLOCK_MONOTONIC_RAW, &end); // set total final time

   suites.clear();
   cout << "==========================================" << endl;
   cout << "CORRECT: " << CORRECT << endl;
   cout << "INCORRECT: " << INCORRECT << endl;
   cout << "FALSE CORRECT: " << FALSE_CORRECT << endl;
   cout << "FALSE INCORRECT: " << FALSE_INCORRECT << endl;
   cout << "NOT SUPPORTED: " << NOT_SUPPORTED << endl;
   cout << "TOLTAL TIME: " << (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/BILLION << "s" << endl;
   cout << "==========================================" << endl;
}
