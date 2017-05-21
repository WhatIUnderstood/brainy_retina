#ifndef FILESYSTEMTOOLS_H
#define FILESYSTEMTOOLS_H

#include <string>
#ifdef __linux__
#include <limits.h>
#include <stdlib.h>
#include <syscall.h>
#endif


class FileSystem{
public:
    static std::string getAbsolutePath(std::string _path)
    {
        std::string path;
        char *full_path = realpath(_path.c_str(), NULL);
        if(full_path != NULL){
            path = std::string(full_path);
            free(full_path);
        }
        return path;
    }

    static inline bool exists (const std::string& name) {
        if (FILE *file = fopen(name.c_str(), "r")) {
            fclose(file);
            return true;
        } else {
            return false;
        }
    }

    static inline std::string getDirName (const std::string& path) {//TODO ., .. folders
        std::string name;
        bool done = false;
        for(int i=path.size()-1; i<=0 && !done;i++){
            char c = path.at(i);
            if(c=='/'
                    || c=='\\'){
                done = true;
                name = path.substr(i);
            }
        }
        if(!done)
            name = path;
        return name;

    }

    static inline bool createPath(std::string path){
        std::string sstring("mkdir -p "+path);
        return system(sstring.c_str());
    }
};
#endif // FILESYSTEMTOOLS_H
