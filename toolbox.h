#ifndef __TOOLBOX_CLASS_H__
#define __TOOLBOX_CLASS_H__

#include <gtkmm.h>
#include "universe_class.h"

#define MAIN_WIN_MIN_HEIGHT 140
#define MAX_UNIVERSE_COUNT 1

class UNIVERSE_ARRAY
{
  public:
    UNIVERSE *dicom;
};


class TOOLBOX : public Gtk::Window
{
  public:
    TOOLBOX();
    virtual ~TOOLBOX();

  protected:
    // Signal Handlers
    void toggle_contour_display_box();
    void remove_all_contours();
    void toggle_dose_overlay();
    void toggle_render_window();
    void set_active_universe( int u );

    void file_close_directory();
    void file_open_directory();
    void file_print_directory();
    void delete_event();

    // Child Widgets
    Gtk::Box        viewBox;
    Gtk::Label      cwdLabel;
    Gtk::Box        mainBox;
    Gtk::Label      label;
    Gtk::Box        cwdBox;

    Gtk::Widget *menuBar;
    Gtk::Widget *toolBar;

    Glib::RefPtr<Gtk::UIManager> manager;
    Glib::RefPtr<Gtk::ActionGroup> actionGroup;

    // Variables
    UNIVERSE_ARRAY universe[2];
    int active_universe;
    int universe_count;
};


#endif // __TOOLBOX_CLASS_H__
