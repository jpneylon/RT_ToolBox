
#include "toolbox.h"

TOOLBOX::TOOLBOX() :
    mainBox( Gtk::ORIENTATION_VERTICAL ),
    active_universe(0),
    universe_count(0)
{
    /* Create the main window */
    set_title("DICOM Toolbox");
    set_position( Gtk::WIN_POS_CENTER );
    set_size_request(180,180);
    set_resizable(true);
    add( mainBox );

    mainBox.set_border_width(1);

    viewBox.set_orientation(Gtk::ORIENTATION_HORIZONTAL);
    viewBox.set_border_width(1);
    viewBox.set_size_request(-1, -1);

    actionGroup = Gtk::ActionGroup::create();

    // File Sub Menu Items
    actionGroup->add( Gtk::Action::create("MenuFile", "_File") );
    actionGroup->add( Gtk::Action::create("Open", Gtk::Stock::OPEN),
        sigc::mem_fun( *this, &TOOLBOX::file_open_directory) );
    actionGroup->add( Gtk::Action::create("Save", Gtk::Stock::SAVE),
        sigc::mem_fun( *this, &TOOLBOX::file_print_directory) );
    actionGroup->add( Gtk::Action::create("Close", Gtk::Stock::CLOSE),
        sigc::mem_fun( *this, &TOOLBOX::file_close_directory) );
    actionGroup->add( Gtk::Action::create("Quit", Gtk::Stock::QUIT),
        sigc::mem_fun( *this, &TOOLBOX::delete_event) );

    // View Sub Menu Items
    //    actionGroup->add_action( "Remove All Contours",
    //    sigc::mem_fun( *this, &TOOLBOX::remove_all_contours) );

    actionGroup->add( Gtk::Action::create("MenuView", "_View") );
    actionGroup->add( Gtk::ToggleAction::create( "toggle_contour", "Toggle _Contours"),
        sigc::mem_fun( *this, &TOOLBOX::toggle_contour_display_box) );
    actionGroup->add( Gtk::ToggleAction::create( "toggle_dose", "Toggle _Dose"),
        sigc::mem_fun( *this, &TOOLBOX::toggle_dose_overlay) );
    actionGroup->add( Gtk::ToggleAction::create( "toggle_vrender", "Toggle _Render"),
        sigc::mem_fun( *this, &TOOLBOX::toggle_render_window) );

    manager = Gtk::UIManager::create();
    manager->insert_action_group( actionGroup );
    add_accel_group( manager->get_accel_group() );

    try
    {
        Glib::ustring ui_info =
            "<ui>"
            "   <menubar name='MenuBar'>"
            "       <menu action='MenuFile'>"
            "           <menuitem action='Open'/>"
            "           <menuitem action='Save'/>"
            "           <menuitem action='Close'/>"
            "           <separator/>"
            "           <menuitem action='Quit'/>"
            "       </menu>"
            "       <menu action='MenuView'>"
            "           <menuitem action='toggle_contour'/>"
            "           <menuitem action='toggle_dose'/>"
            "           <menuitem action='toggle_vrender'/>"
            "       </menu>"
            "   </menubar>"
            "   <toolbar name='ToolBar'>"
            "       <toolitem action='Open'/>"
            "       <toolitem action='Close'/>"
            "       <toolitem action='Quit'/>"
            "   </toolbar>"
            "</ui>";

        manager->add_ui_from_string(ui_info);
    }
    catch (const Glib::Error &ex)
    {
        std::cerr << "\n Building menus and toolbars failed: " << ex.what();
    }

    menuBar = manager->get_widget("/MenuBar");
    mainBox.pack_start( *menuBar, Gtk::PACK_SHRINK );

    toolBar = manager->get_widget("/ToolBar");
    mainBox.pack_start( *toolBar, Gtk::PACK_SHRINK );

    label.set_text("CWD:");
    cwdLabel.set_text("...");

    cwdBox.set_orientation( Gtk::ORIENTATION_HORIZONTAL );
    cwdBox.set_border_width(0);
    cwdBox.pack_start( label, false, false, 2 );
    cwdBox.pack_start( cwdLabel, false, false, 2);

    mainBox.pack_start( viewBox, true, true, 2);
    mainBox.pack_start( cwdBox,  false, false, 2);

    show_all_children();
}


TOOLBOX::~TOOLBOX()
{
    delete_event();
}


/////////////// Callback Functions to Create and Control Axial Display Windows
void
TOOLBOX::toggle_contour_display_box()
{
    if (universe[active_universe].dicom->get_rtstruct_open())
    {
        if (!universe[active_universe].dicom->get_contours_displayed())
        {
            printf("\n ADDING STRUCTURES\n");

            universe[active_universe].dicom->add_contour_display_box();

            //int win_width, win_height;
            //viewBox.get_size_request( win_width, win_height );
            //viewBox.set_size_request( win_width+256, win_height );
            //get_size_request( win_width, win_height );
            //set_size_request( win_width+256, win_height);
        }
        else
        {
            universe[active_universe].dicom->remove_contour_display_box();

            //int win_width, win_height;
            //viewBox.get_size_request( win_width, win_height );
            //viewBox.set_size_request( win_width-256, win_height );
            //get_size_request( win_width, win_height );
            //set_size_request( win_width-256, win_height);
        }
    }
    show_all_children();
}
void
TOOLBOX::remove_all_contours()
{
    if (universe[active_universe].dicom->get_rtstruct_open())
    {
        universe[active_universe].dicom->remove_all_contours();
    }
    show_all_children();
}
void
TOOLBOX::toggle_dose_overlay()
{
    if (universe[active_universe].dicom->get_rtdose_open())
    {
        if (!universe[active_universe].dicom->get_dose_displayed())
        {
            universe[active_universe].dicom->set_overlay(0);
            universe[active_universe].dicom->update_universe();
        }
        else
        {
            universe[active_universe].dicom->set_overlay(-1);
            universe[active_universe].dicom->update_universe();
        }
    }
    show_all_children();
}
void
TOOLBOX::toggle_render_window()
{
    if (universe[active_universe].dicom->get_dicom_open())
    {
        if (!universe[active_universe].dicom->get_renderer_open())
        {
            universe[active_universe].dicom->create_render_window();

            //int win_width, win_height;
            //viewBox.get_size_request( win_width, win_height );
            //viewBox.set_size_request( 2*win_width, win_height );
        }
        else
        {
            universe[active_universe].dicom->destroy_render_window();

            //int win_width, win_height;
            //viewBox.get_size_request( win_width, win_height );
            //int new_w = win_width/2;
            //viewBox.set_size_request( new_w, win_height );
        }
    }
    show_all_children();
}
void
TOOLBOX::set_active_universe( int u )
{
    int active_universe = u;
    printf("\n Active Universe: %d\n", active_universe);
}
void
TOOLBOX::file_close_directory()
{
    if (universe_count == 2)
    {
        delete universe[active_universe].dicom;

        active_universe = 0;
        universe_count--;

        //cwdLabel.set_text( universe[active_universe].dicom->get_dicom_directory() );
    }
    else if (universe_count == 1)
    {
        delete universe[active_universe].dicom;

        active_universe = 0;
        universe_count = 0;

        cwdLabel.set_text( "..." );
        viewBox.set_size_request(32, 32);
        set_size_request(180, 180);
    }

    show_all_children();
}
void
TOOLBOX::file_open_directory()
{
    if (universe_count >= 2)
    {
        printf("\n Two DICOM series already open.\n");
        return;
    }
    universe_count++;
    printf("\n Universe Count: %d\n", universe_count );
    set_active_universe( universe_count - 1 );

    universe[active_universe].dicom = new UNIVERSE;
    universe[active_universe].dicom->open_dicom_directory( universe_count );
    viewBox.pack_start( *universe[active_universe].dicom );

    cwdLabel.set_text( universe[active_universe].dicom->get_dicom_directory() );
    show_all_children();
}
void
TOOLBOX::file_print_directory()
{
    universe[active_universe].dicom->print_dicom_directory();
}
void
TOOLBOX::delete_event()
{
    for (int u=0; u<universe_count; u++)
        delete universe[u].dicom;
    active_universe = 0;
    universe_count = 0;
    hide();
}



