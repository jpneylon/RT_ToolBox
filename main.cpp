
#include "toolbox.h"
#include <gtkmm/application.h>


int main (int argc, char *argv[])
{
    Glib::RefPtr<Gtk::Application> app = Gtk::Application::create( argc, argv, "jacko.toolbox" );

    TOOLBOX toolbox;

    return app->run(toolbox);
}


