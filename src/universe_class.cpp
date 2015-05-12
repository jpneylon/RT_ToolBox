#include "universe_class.h"

EVENTBOX::EVENTBOX( const guint8 *pixelData, int size_x, int size_y, float minVal, float maxVal )
{
    set_events(    Gdk::BUTTON_PRESS_MASK
                 | Gdk::BUTTON_RELEASE_MASK
                 | Gdk::SCROLL_MASK
                 | Gdk::SMOOTH_SCROLL_MASK
                 | Gdk::POINTER_MOTION_MASK);

    pixbuf = Gdk::Pixbuf::create( Gdk::COLORSPACE_RGB,
                                  false, 8,
                                  size_x,
                                  size_y );

    toggleDose.set_label( "Toggle Dose Overlay" );
    toggleDose.signal_activate().connect( sigc::mem_fun( *this, &EVENTBOX::toggle_dose_overlay_flag) );
    popupMenu.append( toggleDose );

    toggleDoseSmoothing.set_label( "Toggle Dose Smoothing" );
    toggleDoseSmoothing.signal_activate().connect( sigc::mem_fun( *this, &EVENTBOX::toggle_dose_smoothing_flag) );
    popupMenu.append( toggleDoseSmoothing );

    popupMenu.show_all();
    popupMenu.accelerate( *this );

    scale = 1;
    sizeX = size_x;
    sizeY = size_y;

    level = 0;
    window = 1024;
    breadth = maxVal - minVal + 1;

	imgFocusX = sizeX/2;
	imgFocusY = sizeY/2;

    overlayDoseFlag = false;
    doseSmoothingFlag = false;

    pixel_data = new guint8[ 3 * size_x * size_y ];
    overlay_data = new guint8[ 3 * size_x * size_y ];
    overlay_raw = new float[ size_x * size_y ];

    red = make_float3(1, 0, 0);
    orange = make_float3(1, 0.64706, 0);
    yellow = make_float3(1, 1, 0);
    green = make_float3(0, 1, 0);
    teal = make_float3(0, 1, 1);
    blue = make_float3(0, 0, 1);
    purple = make_float3(0.62745, 0.1255, 0.9412);
    black = make_float3(0, 0, 0);

    set_pixel_data( pixelData, size_x, size_y );
};
EVENTBOX::~EVENTBOX()
{
    delete [] pixel_data;
    delete [] overlay_data;
    delete [] overlay_raw;
    hide();
};
bool
EVENTBOX::on_draw( const Cairo::RefPtr<Cairo::Context>& cr )
{
    Gdk::Cairo::set_source_pixbuf( cr, pixbuf, 0, 0 );
    cr->paint();
    return false;
}
bool
EVENTBOX::on_scroll_event(GdkEventScroll *event)
{
	// Compute the new scale according to mouse scroll
	double newScale=scale*(1-event->delta_y/20);
	if (newScale < 0.1) newScale = 0.1;

	// Update the center of the image
	double DeltaX=event->x - get_allocated_width()/2.;
	double DeltaY=event->y - get_allocated_height()/2.;
	imgFocusX = imgFocusX + DeltaX/scale - DeltaX/newScale ;
	imgFocusY = imgFocusY + DeltaY/scale - DeltaY/newScale ;;

	// Update scale and redraw the widget
	scale=newScale;
	update_pixbuf();

	// Event has been handled
	return true;
}
bool
EVENTBOX::on_button_press_event(GdkEventButton *event)
{
	// Check if the event is a left button click.
	if (event->button == 1)
	{
		lastX = event->x;
		lastY = event->y;

		return true;
	}
	// right button click
	if (event->button == 3)
	{
        lastX = event->x;
        lastY = event->y;
        popupMenu.popup( event->button, event->time );

        return true;
	}

	return false;
}
bool
EVENTBOX::on_button_release_event(GdkEventButton *event)
{
	// Check if it is the left button
	if (event->button==1)
	{
		//imgFocusX -= (event->x-lastX)/scale;
		//imgFocusY -= (event->y-lastY)/scale;
		update_pixbuf();
		return true;
	}
	return false;
}
bool
EVENTBOX::on_motion_notify_event (GdkEventMotion *event)
{
	// If the left button is pressed, move the view
	if (event->state == GDK_BUTTON1_MASK)
	{
		// Get mouse coordinates
		int XMouse = event->x;
		int YMouse = event->y;

		// Update image focus
		imgFocusX -= (XMouse-lastX)/scale;
		imgFocusY -= (YMouse-lastY)/scale;

		// Memorize new position of the pointer
		lastX = XMouse;
		lastY = YMouse;

		// Update view
		update_pixbuf();
		return true;
	}
	else if (event->state == (GDK_BUTTON1_MASK + GDK_SHIFT_MASK))
	{
        int XMouse = event->x;
        int YMouse = event->y;

        float deltaX = 0.1 * ((float)XMouse - (float)lastX);
        float deltaY = 0.1 * ((float)YMouse - (float)lastY);

        level += deltaX;
        window += deltaY;

        update_pixbuf();
        return true;

	}
	// Event has been handled
	return false;
}
void
EVENTBOX::set_pixel_data( const guint8 *pixelData, int size_x, int size_y )
{
    memcpy( pixel_data, pixelData, 3 * size_x * size_y * sizeof(guint8) );
    update_pixbuf();
}
void
EVENTBOX::set_overlay_max( float x )
{
    overlay_max = x;
}
void
EVENTBOX::toggle_dose_smoothing_flag()
{
    doseSmoothingFlag = !doseSmoothingFlag;
    update_pixbuf();
}
void
EVENTBOX::toggle_dose_overlay_flag()
{
    overlayDoseFlag = !overlayDoseFlag;
    update_pixbuf();
}
void
EVENTBOX::set_overlay_flag( int overlay )
{
    if (overlay == 0)
    {
        overlayDoseFlag = true;
    }
    else
    {
        overlayDoseFlag = false;
    }
}
void
EVENTBOX::set_overlay_data( float *overlayData, int size_x, int size_y )
{
    memcpy( overlay_raw, overlayData, size_x * size_y * sizeof(float) );
    update_pixbuf();
}
float3
EVENTBOX::doseColorLevel(float t)
{
    if (t > 0.95f )
    {
        return(red);
    }
    else if (t > 0.90)
    {
        return(orange);
    }
    else if (t > 0.80)
    {
        return(yellow);
    }
    else if (t > 0.70)
    {
        return(green);
    }
    else if (t > 0.60)
    {
        return(teal);
    }
    else if (t > 0.50)
    {
        return(blue);
    }
    else if (t > 0.25)
    {
        return(purple);
    }
    else
    {
        return(black);
    }
}
float3
EVENTBOX::doseColorRamp(float t)
{
    const int ncolors = 9;
    float c[ncolors][3] =
    {
        { 0.0, 0.0, 0.0, },
        { 0.1, 0.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 1.0, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 0.0, 0.0, },
        { 1.0, 1.0, 1.0, },
    };
    t *= (float)(ncolors-2);
    int i = floor(t);
    float u = t - i;
    float3 rgb;
    rgb.x = lerp(c[i][0], c[i+1][0], u);
    rgb.y = lerp(c[i][1], c[i+1][1], u);
    rgb.z = lerp(c[i][2], c[i+1][2], u);

    return(rgb);
}
void
EVENTBOX::update_pixbuf()
{
    guint8 *imageData = new guint8[ 3 * sizeX * sizeY ];
    memcpy( imageData, pixel_data, 3 * sizeX * sizeY * sizeof(guint8) );

    float minval = 255 * level / breadth;
    float winval = 255 * window / breadth;

    for (int i=0; i < sizeX * sizeY; i++)
    {
        if ( imageData[3*i+0] != imageData[3*i+1] ||
                imageData[3*i+0] != imageData[3*i+2] ||
                   imageData[3*i+1] != imageData[3*i+2] ) continue;
        float val = (float)imageData[3*i];
        val -= minval;
        val /= winval;
        if (val < 0) val = 0;
        if (val > 1) val = 1;
        unsigned char new_val = (unsigned char)(255 * val);
        imageData[3*i+0] = new_val;
        imageData[3*i+1] = new_val;
        imageData[3*i+2] = new_val;
    }

    Glib::RefPtr<Gdk::Pixbuf> display = Gdk::Pixbuf::create_from_data( (const guint8 *)imageData,
                                            Gdk::COLORSPACE_RGB,
                                            false,
                                            8,
                                            sizeX,
                                            sizeY,
                                            sizeX * 3 );

    if (overlayDoseFlag)
    {
        for (int i=0; i < sizeX * sizeY; i++)
        {
            float value = overlay_raw[i] / overlay_max;

            float3 rgb;
            if (doseSmoothingFlag)
                rgb = doseColorRamp(value);
            else
                rgb = doseColorLevel(value);

            overlay_data[3*i+0] = (guint8) 255 * rgb.x;
            overlay_data[3*i+1] = (guint8) 255 * rgb.y;
            overlay_data[3*i+2] = (guint8) 255 * rgb.z;
        }

        Glib::RefPtr<Gdk::Pixbuf> overlay  = Gdk::Pixbuf::create_from_data( (const guint8 *)overlay_data,
                                            Gdk::COLORSPACE_RGB,
                                            false,
                                            8,
                                            sizeX,
                                            sizeY,
                                            sizeX * 3 );

        overlay->composite( display,
                            0, 0,
                            sizeX, sizeY,
                            0, 0,
                            1, 1,
                            Gdk::INTERP_NEAREST,
                            75 );
    }

    int imgWidth = display->get_width();
    int imgHeight = display->get_height();

	pixbuf->fill(0x5F5F5F00);

	double offX = sizeX/2 - imgFocusX*scale;
	double offY = sizeY/2 - imgFocusY*scale;

    double minX = std::max( 0.0, offX );
    double minY = std::max( 0.0, offY );

    double maxX = std::min( (double)sizeX, sizeX/2 + (imgWidth-imgFocusX)*scale);
    double maxY = std::min( (double)sizeY, sizeY/2 + (imgHeight-imgFocusY)*scale);

    double width = maxX - minX;
    double height = maxY - minY;

    display->scale( pixbuf,
                    minX, minY,
                    width, height,
                    offX, offY,
                    scale, scale,
                    Gdk::INTERP_BILINEAR );



    queue_draw();
};



UNIVERSE::UNIVERSE()
{
    set_orientation( Gtk::ORIENTATION_HORIZONTAL );

    dicom_open = false;
    rtdose_open = false;
    rtstruct_open = false;
    rtplan_open = false;

    renderer_open = false;
    dose_displayed = false;
    contours_displayed = false;

    rx_dose.x = 0;
    rx_dose.y = 0;
    rx_dose.z = 0;

    slice = 0;

    isocenter.x = 999999;
    isocenter.y = 999999;
    isocenter.z = 999999;
}

UNIVERSE::~UNIVERSE()
{
    close_dicom_directory();
    hide();
}

void
UNIVERSE::set_overlay( int i )
{
    overlay = i;
    if (overlay == 0)
    {
        dose_displayed = true;
        eventbox->set_overlay_max( rx_dose.x );
    }
    else
        dose_displayed = false;
};
void
UNIVERSE::update_buff()
{
    int index = slice;

    if (overlay == 0)
    {
        int offset = slice * upscaler * dose->size.x * upscaler * dose->size.y;
        eventbox->set_overlay_data( dose->matrix + offset,
                                    upscaler  * dose->size.x,
                                    upscaler  * dose->size.y );
    }
    eventbox->set_pixel_data( (const guint8 *) images->anatomy[index].pixels,
                                               upscaler * grid->size.x,
                                               upscaler * grid->size.y );
}
void
UNIVERSE::plus_buff()
{
    slice = abs(slice + 1) % grid->size.z;

    slice_adjust->set_value( (double)slice );
    update_buff();
}
void
UNIVERSE::minus_buff()
{
    if (slice > 0)
        slice = abs(slice - 1) % grid->size.z;
    else
        slice = grid->size.z - 1;

    slice_adjust->set_value( (double)slice );
    update_buff();
}
void
UNIVERSE::update_slice()
{
    slice = (int) slice_adjust->get_value();
    //printf("\n Slice %d",slice);
    update_buff();
}
void
UNIVERSE::make_DisplayWindow()
{
    Gtk::Box    *vbox;
    Gtk::Box    *labelbox;
    Gtk::Box    *h_button_box;
    Gtk::Box    *h_img_box;

    Gtk::Button *minus_button;
    Gtk::Button *plus_button;

    Gtk::Scale  *slice_scale;

    Gtk::Label  *label;

    eventbox = new EVENTBOX( (const guint8*) images->anatomy[slice].pixels,
                                             upscaler * grid->size.x,
                                             upscaler * grid->size.y,
                                             grid->min, grid->max );

    scroll_win = new Gtk::ScrolledWindow();
    scroll_win->set_policy( Gtk::POLICY_AUTOMATIC, Gtk::POLICY_AUTOMATIC );
    scroll_win->add( *eventbox );
    scroll_win->set_size_request( grid->size.x+WIN_BUF, grid->size.y+WIN_BUF );

    //Create label box to display current kV slice
    label = new Gtk::Label( "Slice" );

    //Create button to display previous slice
    minus_button = new Gtk::Button();
    minus_button->set_label("Previous");
    minus_button->signal_clicked().connect( sigc::mem_fun( *this, &UNIVERSE::minus_buff) );

    //Create button to display next slice
    plus_button = new Gtk::Button();
    plus_button->set_label("Next");
    plus_button->signal_clicked().connect( sigc::mem_fun( *this, &UNIVERSE::plus_buff) );

    //Create Range Widgets
    slice_adjust = Gtk::Adjustment::create( slice, 0, grid->size.z - 1, 1, 1, 1);
    slice_adjust->signal_value_changed().connect( sigc::mem_fun( *this, &UNIVERSE::update_slice) );

    slice_scale = new Gtk::Scale( slice_adjust, Gtk::ORIENTATION_HORIZONTAL );
    slice_scale->set_digits(0);

    //Pack widgets
    vbox         = new Gtk::Box(Gtk::ORIENTATION_VERTICAL,     10);
    labelbox     = new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL,   5);
    h_img_box    = new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL,   5);
    h_button_box = new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL,   5);

    labelbox->pack_start( *label, false, false, 0);
    labelbox->pack_start( *slice_scale, true, true, 0);

    h_img_box->pack_start( *scroll_win, true, true, 0);

    h_button_box->pack_start( *minus_button, true, true, 0);
    h_button_box->pack_start( *plus_button, true, true, 0);

    vbox->pack_start( *h_img_box, true, true, 0);
    vbox->pack_start( *labelbox, false, false, 0);
    vbox->pack_start( *h_button_box, false, false, 0);
    vbox->show();

    pack_start( *vbox, true, true, 3);
    show_all_children();
}
void
UNIVERSE::update_universe()
{
    eventbox->set_overlay_flag( overlay );
    update_buff();
}


float
UNIVERSE::luminosity_contrast(int3 background, int3 text)
{
    float L1 = 0.2126 * pow( (float)background.x / 255, 2.2) +
                0.7152 * pow( (float)background.y / 255, 2.2) +
                0.0722 * pow( (float)background.z / 255, 2.2);

    float L2 = 0.2126 * pow( (float)text.x / 255, 2.2) +
                0.7152 * pow( (float)text.y / 255, 2.2) +
                0.0722 * pow( (float)text.z / 255, 2.2);

    if ( L1 > L2 )
        return (L1+0.05)/(L2+0.05);
    else
        return (L2+0.05)/(L1+0.05);
}
void
UNIVERSE::set_contour_button_css_styling(char *name, Gtk::ToggleButton *button, int3 rgb )
{
    int text_color = 0;
    int3 black; black.x = black.y = black.z = 0;
    int3 white; white.x = white.y = white.z = 255;
    float black_text_contrast = luminosity_contrast( rgb, black );
    float white_text_contrast = luminosity_contrast( rgb, white );
    if (white_text_contrast > black_text_contrast)
        text_color = 255;

    char css_temp[1024];
    sprintf( css_temp ," .%s { color: rgb(%d,%d,%d); background: rgb(%d,%d,%d); }",
                                                                    name,
                                                                    text_color,
                                                                    text_color,
                                                                    text_color,
                                                                    rgb.x,
                                                                    rgb.y,
                                                                    rgb.z);
    std::string css_text = css_temp;

    Glib::RefPtr<Gtk::CssProvider> provider = Gtk::CssProvider::create();
    provider->load_from_data( css_text );

    Glib::RefPtr<Gtk::StyleContext> style = button->get_style_context();
    style->add_provider( provider, GTK_STYLE_PROVIDER_PRIORITY_USER );
    style->add_class(name);
}
void
UNIVERSE::copy_structure_data( int idx )
{
    structures->object[idx].subCntrs = plan_struct->getROISubCntrCount(idx);
    structures->object[idx].TOTpoints = plan_struct->getROITotalPointsCount(idx);

    structures->object[idx].CTRpoints = new int[ structures->object[idx].subCntrs ];
    structures->object[idx].matrix = new float[ 3 * structures->object[idx].TOTpoints ];

    uint point_count = 0;
    for (int s=0; s<structures->object[idx].subCntrs; s++)
    {
        structures->object[idx].CTRpoints[s] = plan_struct->getROISubCntrPointCount(idx,s);
        memcpy( structures->object[idx].matrix + point_count,
                plan_struct->getROISubCntrPoints(idx,s),
                3*structures->object[idx].CTRpoints[s]*sizeof(float) );
        point_count += 3*structures->object[idx].CTRpoints[s];
    }
}
void
UNIVERSE::draw_contour(const Glib::ustring &label)
{
    int idx = atoi( label.data() );

    if (loaded_contour[idx] == 0)
    {
        loaded_contour[idx] = plan_struct->loadRTStructData(idx);
    }

    if (loaded_contour[idx] == 1)
    {
        copy_structure_data(idx);
        structures->object[idx].draw = !structures->object[idx].draw;
        if (structures->object[idx].draw)
            add_structure_contours( grid,
                                    images,
                                    &structures->object[idx],
                                    upscaler);
        else if (!structures->object[idx].draw)
            remove_structure_contours( grid,
                                       images,
                                       &structures->object[idx],
                                       upscaler);
    }
    update_buff();
}
void
UNIVERSE::add_contour_display_box ()
{
    if (rtstruct_open)
    {
        printf("\n ADDING STRUCTURES\n");

        color_scroll = new Gtk::ScrolledWindow();
        color_scroll->set_policy( Gtk::POLICY_AUTOMATIC, Gtk::POLICY_AUTOMATIC );

        Gtk::Box *color_box = new Gtk::Box( Gtk::ORIENTATION_VERTICAL, 1 );
        color_scroll->add( *color_box );
        color_scroll->set_size_request(200, -1);

        for (int c=0; c < structures->CTRnumber; c++)
        {
            structures->object[c].draw = false;

            Gtk::ToggleButton *color;
            color = new Gtk::ToggleButton();

            char temp[100];
            sprintf(temp,"%d - %s", c, structures->object[c].ROIname);
            color->set_label( temp );

            char name_temp[16];
            sprintf(name_temp,"button_%d",c);
            color->set_name( name_temp );

            /*Gdk::RGBA rgba;
            rgba.set_rgba_u( structures->object[c].rgb.x,
                             structures->object[c].rgb.y,
                             structures->object[c].rgb.z,
                             1.0 );
            color->override_background_color( rgba );*/
            set_contour_button_css_styling( name_temp, color, structures->object[c].rgb);

            color->signal_clicked().connect( sigc::bind<Glib::ustring>( sigc::mem_fun( *this, &UNIVERSE::draw_contour), temp ) );

            color_box->pack_start( *color, true, true, 0 );
        }
        pack_start( *color_scroll, false, false, 3 );
        show_all_children();

        contours_displayed = true;
    }
}
void
UNIVERSE::remove_contour_display_box ()
{
    if (rtstruct_open)
    {
        remove( *color_scroll );
        show_all_children();

        contours_displayed = false;
    }
    update_buff();
}
void
UNIVERSE::remove_all_contours ()
{
    if (rtstruct_open)
    {
        // remove any contours from image
        for (int c=0; c<structures->CTRnumber; c++)
            if (loaded_contour[c] == 1)
                if (structures->object[c].draw)
                {
                    structures->object[c].draw = false;
                    remove_structure_contours( grid,
                                               images,
                                               &structures->object[c],
                                               upscaler);
                }
    }
    update_buff();
}


void
UNIVERSE::reorder_data( FLOAT_GRID *dens )
{
    printf("\n Reordering Data...\n");
    fflush(stdout);

    int num_slices = dens->size.z;
    int slice_size = dens->size.x*dens->size.y;

    float *temp_data;
    temp_data = new float[slice_size];

    for(int k=0; k<num_slices/2; k++)
    {
        memcpy(temp_data, dens->matrix+k*slice_size, slice_size*sizeof(float));
        memcpy(dens->matrix+k*slice_size, dens->matrix+(num_slices-1-k)*slice_size, slice_size*sizeof(float));
        memcpy(dens->matrix+(num_slices-1-k)*slice_size, temp_data, slice_size*sizeof(float));
    }

    delete [] temp_data;
}


void
UNIVERSE::close_dicom_directory()
{
    if (rtplan_open)
    {
        delete plan_plan;
        rtplan_open = false;
    }
    if (rtstruct_open)
    {
        delete [] loaded_contour;
        delete structures;
        delete plan_struct;
        rtstruct_open = false;
    }
    if (rtdose_open)
    {
        for (int k=0; k<grid->size.z; k++)
            delete [] images->overlay[k].pixels;
        delete plan_dose;
        rtdose_open = false;
    }
    if (renderer_open)
    {
        destroy_render_window();
        renderer_open = false;
    }

    for (int k=0; k<grid->size.z; k++)
        delete [] images->anatomy[k].pixels;

    delete eventbox;

    delete dose;
    delete images;
    delete grid;
    dicom_open = false;
    contours_displayed = false;
    dose_displayed = false;
    universe = -1;
    hide();


}
void
UNIVERSE::select_dicom_directory()
{
    GtkWidget *file;

    file = gtk_file_chooser_dialog_new( title, NULL,
                                        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
                                        "CANCEL", GTK_RESPONSE_CANCEL,
                                        "OPEN", GTK_RESPONSE_ACCEPT,
                                        NULL);
    gtk_file_chooser_set_filename( GTK_FILE_CHOOSER (file), STARTDIR );

    if (gtk_dialog_run ( GTK_DIALOG (file)) == GTK_RESPONSE_ACCEPT)
    {
        dicomdir = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (file));
    }
    gtk_widget_destroy (file);
}
void
UNIVERSE::open_dicom_directory ( int universe_count )
{
    ///////////////////////////////////////// Select Planning CT /////////////////////////////////
    sprintf( title, "Planning CT" );
    universe = universe_count - 1;

    select_dicom_directory();

    plan_ct = new RTImage();
    plan_ct->setDicomDirectory( dicomdir );
    printf("\n Plan_CT Dicom Dir: %s",  plan_ct->getDicomDirectory());

    if ( plan_ct->loadDicomInfo() )
    {
        dicom_open = true;
        plan_ct->loadRTImageData();

        grid = new FLOAT_GRID;
        grid->size =  plan_ct->getDataSize();
        grid->voxel =  plan_ct->getVoxelSize();
        grid->startPos =  plan_ct->getSliceImagePositionPatient( 0 );
        printf("\n DATA START Pos: %2.3f x %2.3f x %2.3f ",  grid->startPos.x,
               grid->startPos.y,
               grid->startPos.z);
        grid->endPos =  plan_ct->getSliceImagePositionPatient(  grid->size.z-1 );
        printf("\n DATA END Pos: %2.3f x %2.3f x %2.3f\n\n",  grid->endPos.x,
               grid->endPos.y,
               grid->endPos.z);
        grid->min =  plan_ct->getDataMin();
        grid->max =  plan_ct->getDataMax();
        grid->matrix =  plan_ct->getDataArray();
        grid->window =  grid->max -  grid->min;
        grid->level =  plan_ct->getWindowCenter();

        grid->bracket.x =  grid->min;
        grid->bracket.y =  grid->max;

        grid->shift.x = 0;
        grid->shift.y = 0;
        grid->shift.z = 0;
        grid->shift.w = 0;

        plan_struct = new RTStruct;
        plan_struct->setDicomDirectory(  dicomdir );
        if (  plan_struct->loadDicomInfo() )
        {
            structures = new STRUCT_SET;

            plan_struct->loadRTStructInfo();

            rtstruct_open = true;
            structures->CTRnumber =  plan_struct->getNumberOfROIs();

            for (int c=0; c< structures->CTRnumber; c++)
            {
                sprintf(  structures->object[c].ROIname, "%s",  plan_struct->getROIName(c) );
                structures->object[c].rgb =  plan_struct->getROIColor(c);
                //int3 rgb =  plan_struct->getROIColor(c);
                //structures->object[c].rgb.x =  (float)rgb.x / 255.f;
                //structures->object[c].rgb.y =  (float)rgb.y / 255.f;
                //structures->object[c].rgb.z =  (float)rgb.z / 255.f;
                structures->object[c].draw = false;
                structures->object[c].ROInumber =  plan_struct->getROINumber(c);
            }

            loaded_contour = new int[ structures->CTRnumber];
            memset( loaded_contour,0, structures->CTRnumber*sizeof(int));
        }
        else printf("\n Moving on from RTSTRUCT...\n");

        plan_plan = new RTPlan();
        plan_plan->setDicomDirectory(  dicomdir );
        if (  plan_plan->loadDicomInfo() )
        {
            rtplan_open = true;
            plan_plan->loadRTPlanData();
            if ( rx_dose.x == 0.f &&
                    rx_dose.y == 0.f &&
                    rx_dose.z == 0.f)
            {
                rx_dose =  plan_plan->getRXDoseLevels();
                //total_fractions = plan_plan->getFractionCount();
            }
            if ( isocenter.x == 999999 &&
                    isocenter.y == 999999 &&
                    isocenter.z == 999999)
            {
                isocenter =  plan_plan->getIsocenter();
                printf("\n ISOCENTER: %f x %f x %f\n\n",  isocenter.x,
                       isocenter.y,
                       isocenter.z);
            }
            //else
            //    dicom->grid.isocenter = ISOCENTER;
        }
        else printf("\n Moving on from RTPLAN...\n");

        plan_dose = new RTDose();
        plan_dose->setDicomDirectory(  dicomdir );
        if (  plan_dose->loadDicomInfo() )
        {
            dose = new FLOAT_GRID;
            plan_dose->loadRTDoseData();

            dose->size =  plan_dose->getDataSize();
            dose->voxel =  plan_dose->getVoxelSize();
            dose->min =  plan_dose->getDataMin();
            dose->max =  plan_dose->getDataMax();
            dose->startPos =  plan_dose->getDataOrigin();
            dose->matrix =  plan_dose->getDataArray();

            max_dose =  plan_dose->getDataMax();
            if ( rx_dose.x == 0.f &&
                    rx_dose.y == 0.f &&
                      rx_dose.z == 0.f)
                rx_dose.x =  max_dose;
            printf("\n MAX_DOSE: %f\n RX_DOSE: %f x %f x %f\n",  max_dose,
                   rx_dose.x,
                   rx_dose.y,
                   rx_dose.z );

            if ( fabs( grid->endPos.z -  dose->startPos.z) <
                    fabs( grid->startPos.z -  dose->startPos.z) )
            {
                reorder_data( dose );
                dose->startPos.z += ((float) dose->size.z *
                                     dose->voxel.z);
                printf("\n DATA Position: %2.3f x %2.3f x %2.3f ",  dose->startPos.x,
                       dose->startPos.y,
                       dose->startPos.z);
            }

            if (  grid->size.x !=  dose->size.x ||
                    grid->size.y !=  dose->size.y ||
                      grid->size.z !=  dose->size.z )
            {
                resize_dose_data(  grid,  dose );
            }
            rtdose_open = true;
        }
        else printf("\n Moving on from RTDOSE...\n");

        upscaler = UPSCALE;
        float3 rgb;
        rgb.x = 1;
        rgb.y = 1;
        rgb.z = 1;

        overlay = -1;
        images = new IMAGE_SET;
        images->anatomy = new ILIST[ grid->size.z];
        createAxialImageSet(  grid,
                              images,
                              rgb,
                              upscaler);

        make_DisplayWindow();
    }
    // else { pop up error window }
}


void
UNIVERSE::print_dicom_directory()
{
    if (dicom_open)
    {
        printf("\n Current Directory: %s\n", dicomdir);
    }
}
void
UNIVERSE::print_rtstruct_directory()
{
    if (rtstruct_open)
    {
        printf("\n RTSTRUCT Directory: %s\n", plan_struct->getDicomDirectory() );
        fflush(stdout);
    }
    else
    {
        printf("\n No RTSTRUCT File loaded.\n");
    }
}
void
UNIVERSE::print_rtdose_directory()
{
    if (rtdose_open)
    {
        printf("\n RTDOSE Directory: %s\n", plan_dose->getDicomDirectory() );
        fflush(stdout);
    }
    else
    {
        printf("\n No RTDOSE File loaded.\n");
    }
}





void
UNIVERSE::update_render_buffer()
{
    Glib::RefPtr<Gdk::Pixbuf> render_pixbuf = Gdk::Pixbuf::create_from_data((const guint8*) vrender->get_vrender_buffer(),
                                                                                               Gdk::COLORSPACE_RGB,
                                                                                               false,
                                                                                               8,
                                                                                               vrender->get_width(),
                                                                                               vrender->get_height(),
                                                                                               vrender->get_width() * 3 );

    render_image.set( render_pixbuf );
}
void
UNIVERSE::set_render_density()
{
    float r_dens = (float) dens_adjust->get_value();
    vrender->set_density(r_dens);
    update_render_buffer();
}
void
UNIVERSE::set_render_brightness()
{
    float r_bright = (float) bright_adjust->get_value();
    vrender->set_brightness(r_bright);
    update_render_buffer();
}
void
UNIVERSE::set_render_offset()
{
    float r_offset = (float) offset_adjust->get_value();
    vrender->set_offset(r_offset);
    update_render_buffer();
}
void
UNIVERSE::set_render_scale()
{
    float r_scale = (float) scale_adjust->get_value();
    vrender->set_scale(r_scale);
    update_render_buffer();
}

void
UNIVERSE::update_render_zoom(gdouble x, gdouble y)
{
    float dy = (y - vrender->get_last_y()) / 100;

    vrender->set_vrender_zoom( dy );
    update_render_buffer();

    vrender->set_last_x(x);
    vrender->set_last_y(y);
}
void
UNIVERSE::update_render_translation(gdouble x, gdouble y)
{
    float dx = (x - vrender->get_last_x()) / 100;
    float dy = (y - vrender->get_last_y()) / 100;

    vrender->set_vrender_translation( dx, dy );
    update_render_buffer();

    vrender->set_last_x(x);
    vrender->set_last_y(y);
}
void
UNIVERSE::update_render_rotation(gdouble x, gdouble y)
{
    //printf("\n Event: %f %f",x,y);
    float dx = (x - vrender->get_last_x()) / 5;
    float dy = (y - vrender->get_last_y()) / 5;

    vrender->set_vrender_rotation( dx, dy );
    update_render_buffer();

    vrender->set_last_x(x);
    vrender->set_last_y(y);
}
bool
UNIVERSE::render_button_press_event(GdkEventButton *event)
{
    vrender->set_last_x( event->x );
    vrender->set_last_y( event->y );

    if (event->button == GDK_BUTTON_PRIMARY)
    {
        update_render_rotation(event->x, event->y);
    }
    else if (event->button == GDK_BUTTON_SECONDARY)
    {
        update_render_translation(event->x, event->y);
    }
    else if (event->button == GDK_BUTTON_MIDDLE)
    {
        update_render_zoom(event->x, event->y);
    }
    return true;
}
bool
UNIVERSE::render_motion_notify_event(GdkEventMotion *event)
{
    /*printf("\n Event-State: %d\n Button1-Mask: %d\n Condition1: %d\n Condition2: %d\n Condition3: %d\n",
                event->state, GDK_BUTTON1_MASK,
                event->state == GDK_BUTTON1_MASK,
                event->state == (GDK_BUTTON1_MASK + GDK_SHIFT_MASK),
                event->state == (GDK_BUTTON1_MASK + GDK_CONTROL_MASK) );*/

    if (event->state == GDK_BUTTON1_MASK)
    {
        update_render_rotation(event->x, event->y);
    }
    else if (event->state == (GDK_BUTTON1_MASK + GDK_SHIFT_MASK))
    {
        update_render_translation(event->x, event->y);
    }
    else if (event->state == (GDK_BUTTON1_MASK + GDK_CONTROL_MASK))
    {
        update_render_zoom(event->x, event->y);
    }
    return true;
}
void
UNIVERSE::create_render_window()
{
    vrender = new VRender;
    vrender->init_vrender( grid->matrix, grid->size, grid->max, grid->min, 0 );

    Gtk::Box *render_vbox = new Gtk::Box(Gtk::ORIENTATION_VERTICAL, 1);

    Gtk::ScrolledWindow *render_scroll;
    Gtk::EventBox       *render_eventbox;

    Glib::RefPtr<Gdk::Pixbuf> render_pixbuf;

    Gtk::Scale          *dens_scale;
    Gtk::Box            *dens_hbox;
    Gtk::Label          *dens_label;

    Gtk::Scale          *bright_scale;
    Gtk::Box            *bright_hbox;
    Gtk::Label          *bright_label;

    Gtk::Scale          *offset_scale;
    Gtk::Box            *offset_hbox;
    Gtk::Label          *offset_label;

    Gtk::Scale          *scale_scale;
    Gtk::Box            *scale_hbox;
    Gtk::Label          *scale_label;

    render_eventbox = new Gtk::EventBox();
    render_eventbox->set_events(  Gdk::BUTTON_PRESS_MASK
                                | Gdk::BUTTON_RELEASE_MASK
                                | Gdk::POINTER_MOTION_MASK
                                | Gdk::POINTER_MOTION_HINT_MASK
                                | Gdk::BUTTON_RELEASE_MASK);

    render_pixbuf->create_from_data((const guint8*) vrender->get_vrender_buffer(),
                                                    Gdk::COLORSPACE_RGB,
                                                    false,
                                                    8,
                                                    vrender->get_width(),
                                                    vrender->get_height(),
                                                    vrender->get_width() * 3 );

    render_image.set( render_pixbuf );
    render_eventbox->add( render_image );

    render_eventbox->signal_motion_notify_event().connect( sigc::mem_fun( *this, &UNIVERSE::render_motion_notify_event) );
    render_eventbox->signal_button_press_event().connect( sigc::mem_fun( *this, &UNIVERSE::render_button_press_event) );

    render_scroll = new Gtk::ScrolledWindow();
    render_scroll->set_policy(Gtk::POLICY_AUTOMATIC, Gtk::POLICY_ALWAYS);
    render_scroll->add(render_eventbox[0]);

    //////////// CREATE SLIDERS
    dens_hbox = new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL, 1);
    dens_label = new Gtk::Label("Density: ");
    dens_hbox->pack_start(dens_label[0], false, false, 0);
    dens_adjust = Gtk::Adjustment::create( vrender->get_density(), 0, 1.1, 0.01, 0.1, 0.1);
    dens_adjust->signal_value_changed().connect( sigc::mem_fun( *this, &UNIVERSE::set_render_density) );
    dens_scale = new Gtk::Scale( dens_adjust, Gtk::ORIENTATION_HORIZONTAL );
    dens_scale->set_digits(2);
    dens_hbox->pack_start(dens_scale[0], true, true, 0);
    render_vbox->pack_start(dens_hbox[0], false, false, 0);

    bright_hbox = new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL, 1);
    bright_label = new Gtk::Label("Brightness: ");
    bright_hbox->pack_start(bright_label[0], false, false, 0);
    bright_adjust = Gtk::Adjustment::create( vrender->get_brightness(), 0, 5.1, 0.01, 0.1, 0.1);
    bright_adjust->signal_value_changed().connect( sigc::mem_fun( *this, &UNIVERSE::set_render_brightness) );
    bright_scale = new Gtk::Scale( bright_adjust, Gtk::ORIENTATION_HORIZONTAL );
    bright_scale->set_digits(2);
    bright_hbox->pack_start(bright_scale[0], true, true, 0);
    render_vbox->pack_start(bright_hbox[0], false, false, 0);

    render_vbox->pack_start(render_scroll[0], true, true, 0);

    offset_hbox = new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL, 1);
    offset_label = new Gtk::Label("Offset: ");
    offset_hbox->pack_start(offset_label[0], false, false, 0);
    offset_adjust = Gtk::Adjustment::create( vrender->get_offset(), -1, 1.1, 0.01, 0.1, 0.1);
    offset_adjust->signal_value_changed().connect( sigc::mem_fun( *this, &UNIVERSE::set_render_offset) );
    offset_scale = new Gtk::Scale( offset_adjust, Gtk::ORIENTATION_HORIZONTAL );
    offset_scale->set_digits(2);
    offset_hbox->pack_start(offset_scale[0], true, true, 0);
    render_vbox->pack_start(offset_hbox[0], false, false, 0);

    scale_hbox = new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL, 1);
    scale_label = new Gtk::Label("Scale: ");
    scale_hbox->pack_start(scale_label[0], false, false, 0);
    scale_adjust = Gtk::Adjustment::create( vrender->get_scale(), 0, 5.5, 0.1, 0.5, 0.5);
    scale_adjust->signal_value_changed().connect( sigc::mem_fun( *this, &UNIVERSE::set_render_scale) );
    scale_scale = new Gtk::Scale( scale_adjust, Gtk::ORIENTATION_HORIZONTAL );
    scale_scale->set_digits(2);

    scale_hbox->pack_start(offset_scale[0], true, true, 0);
    render_vbox->pack_start(scale_hbox[0], false, false, 0);

    pack_start(render_vbox[0], true, true, 0);
    show_all_children();

    renderer_open = true;
}
void
UNIVERSE::destroy_render_window()
{
    delete vrender;
    renderer_open = false;
}
