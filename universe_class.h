#ifndef __UNIVERSE_CLASS_H__
#define __UNIVERSE_CLASS_H__

#include "RTClasses/rtimages.h"
#include "RTClasses/rtdose.h"
#include "RTClasses/rtstruct.h"
#include "RTClasses/rtplan.h"

#include "vrender_class.h"
#include "defs.h"

#include <gtkmm.h>

#define STARTDIR "/media/jacko/SHAREDRIVEA/PtData/test/kVCT/"
#define UPSCALE 1
#define WIN_BUF 5

/////////// Functions to create Axial Slice Display Windows
int resize_dose_data( FLOAT_GRID *, FLOAT_GRID * );
int createAxialImageSet(FLOAT_GRID *, IMAGE_SET *, float3, int );
int createSagittalImageSet(FLOAT_GRID *, IMAGE_SET *, float3, int );
int createCoronalImageSet(FLOAT_GRID *, IMAGE_SET *, float3, int );
int createOrientationImage(FLOAT_GRID *, IMAGE_SET *, float3, int );
int createOverlayImageSet(FLOAT_GRID *, IMAGE_SET *, float, int, int );
int add_structure_contours( FLOAT_GRID *, IMAGE_SET *, CNTR_SPECS *, int );
int remove_structure_contours( FLOAT_GRID *, IMAGE_SET *, CNTR_SPECS *, int );


class EVENTBOX : public Gtk::EventBox
{
  public:
    EVENTBOX( const guint8 *pixelData, int size_x, int size_y, float minVal, float maxVal );
    virtual ~EVENTBOX();

    void set_pixel_data( const guint8 *pixelData, int size_x, int size_y );
    //void set_overlay_data( const guint8 *overlayData, int size_x, int size_y );
    void set_overlay_data( float *overlayData, int size_x, int size_y );
    void set_overlay_flag( int overlay );
    void set_overlay_max( float x );

  protected:
    virtual bool on_draw( const Cairo::RefPtr<Cairo::Context> &cr );
    void toggle_dose_smoothing_flag();
    void toggle_dose_overlay_flag();
    bool on_scroll_event(GdkEventScroll *event);
    bool on_button_press_event(GdkEventButton *event);
    bool on_button_release_event(GdkEventButton *event);
    bool on_motion_notify_event (GdkEventMotion *event);
    void update_pixbuf();

    float3 doseColorRamp(float t);
    float3 doseColorLevel(float t);

    guint8 *pixel_data;
    guint8 *overlay_data;
    float  *overlay_raw;

    Gtk::Menu       popupMenu;
    Gtk::MenuItem   toggleDose;
    Gtk::MenuItem   toggleDoseSmoothing;
    Gtk::SeparatorMenuItem menuItemLine;

    Glib::RefPtr<Gdk::Pixbuf> pixbuf;
    double scale;
    double imgFocusX, imgFocusY;
    int lastX, lastY;
    int sizeX, sizeY;

    bool overlayDoseFlag;
    bool doseSmoothingFlag;

    float level, window, breadth;
    float overlay_max;
    float3 red, orange, yellow, green, teal, blue, purple, black;
};


class UNIVERSE : public Gtk::Box
{
  public:
    UNIVERSE();
    virtual ~UNIVERSE();

    void open_dicom_directory( int universe_count );
    void close_dicom_directory();

    void print_dicom_directory();
    void print_rtdose_directory();
    void print_rtstruct_directory();

    void create_render_window();
    void destroy_render_window();

    bool get_renderer_open(){ return renderer_open; };
    bool get_dose_displayed(){ return dose_displayed; };
    bool get_contours_displayed(){ return contours_displayed; };

    bool get_dicom_open(){ return dicom_open; };
    bool get_rtstruct_open(){ return rtstruct_open; };
    bool get_rtdose_open(){ return rtdose_open; };
    bool get_rtplan_open(){ return rtplan_open; };

    void set_overlay( int i );
    void update_universe();

    void add_contour_display_box();
    void remove_contour_display_box();
    void remove_all_contours();

    char *get_dicom_directory(){ return dicomdir; };

  private:

    void set_active_universe( int univ );
    void plus_buff();
    void minus_buff();
    void update_slice();
    void update_buff();
    void make_DisplayWindow();

    void select_dicom_directory();
    void reorder_data( FLOAT_GRID *data );

    void update_render_buffer();
    void set_render_density();
    void set_render_brightness();
    void set_render_offset();
    void set_render_scale();
    void update_render_zoom(gdouble x, gdouble y);
    void update_render_translation(gdouble x, gdouble y);
    void update_render_rotation(gdouble x, gdouble y);
    virtual bool render_button_press_event(GdkEventButton *event);
    virtual bool render_motion_notify_event(GdkEventMotion *event);

    float luminosity_contrast(int3 background, int3 text);
    void set_contour_button_css_styling(char *name, Gtk::ToggleButton *button, int3 rgb);
    void copy_structure_data( int idx );
    void draw_contour( const Glib::ustring &label );

    RTDose *plan_dose;
    RTStruct *plan_struct;
    RTPlan *plan_plan;
    RTImage *plan_ct;
    RTImage *daily_ct;
    VRender *vrender;

    char *dicomdir;
    char title[16];

    FLOAT_GRID *grid;
    IMAGE_SET *images;
    FLOAT_GRID *dose;
    STRUCT_SET *structures;
    int *loaded_contour;

    bool dicom_open;
    bool rtdose_open;
    bool rtstruct_open;
    bool rtplan_open;

    bool renderer_open;
    bool dose_displayed;
    bool contours_displayed;

    float3 isocenter;
    float3 rx_dose;
    float max_dose;

    int universe;
    int overlay;
    int slice;
    int upscaler;
    bool flip;

    EVENTBOX                *eventbox;
    Gtk::ScrolledWindow     *scroll_win;
    Gtk::ScrolledWindow     *color_scroll;
    Gtk::Image              render_image;

    Glib::RefPtr<Gtk::Adjustment>     slice_adjust;
    Glib::RefPtr<Gtk::Adjustment>     shift_adjust;

    Glib::RefPtr<Gtk::Adjustment>     dens_adjust;
    Glib::RefPtr<Gtk::Adjustment>     bright_adjust;
    Glib::RefPtr<Gtk::Adjustment>     offset_adjust;
    Glib::RefPtr<Gtk::Adjustment>     scale_adjust;
};

#endif // __UNIVERSE_CLASS_H__
