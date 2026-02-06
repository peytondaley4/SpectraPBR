#include "application.h"
#include <iostream>

int main(int argc, char* argv[]) {
    spectra::Application app;

    if (!app.parseArgs(argc, argv)) {
        std::cerr << "Failed to parse arguments\n";
        return 1;
    }

    if (!app.init()) {
        std::cerr << "Failed to initialize application\n";
        return 1;
    }

    app.run();
    app.shutdown();

    return 0;
}
