from fpdf import FPDF


def create_report(insights, results):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_title('Forecast Report')
    pdf.set_author('Hélio Domingos')
    pdf.print_insights_in_data(1, 'Insights in data', insights)
    #pdf.print_chapter(1, 'Insights in data', insights)
    #pdf.print_chapter(2, 'THE PROS AND CONS', 'report/c2.txt')

    pdf.output('report/report.pdf', 'F')
    print("pdf created")
    return 0


class PDF(FPDF):
    def header(self):
        # Logo
        # self.image('logo_pb.png', 10, 8, 33)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Title
        self.cell(0, h=25, txt='Forecast Results Report', border=5, ln=0, align='C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, num, label):
        # Arial 12
        self.set_font('Arial', '', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, 'Chapter %d : %s' % (num, label), 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_body(self, structure):
        # Read text file
        #with open(name, 'rb') as fh:
        #    txt = fh.read().decode('latin-1')
        # Times 12
        self.set_font('Arial', '', 12)
        # Output justified text
        self.cell(0, 5, "Initial data plot")
        #self.cell(0,structure['initial_data_plot'])# Line break

        #self.image(structure["initial_data_plot"], x = 0, y = 50, w = 200, h = 50,)
        #self.ln()
        self.image("report/moving_average_5_raw_to_month.png", x=0, y=50, w=200, h=50, )
        self.ln()
        # Mention in italics
        #self.set_font('', 'I')
        #self.cell(0, 5, '(end of excerpt)')

    def print_chapter(self, num, title, name):
        self.add_page()
        self.chapter_title(num, title)
        self.chapter_body(name)

    def print_insights_in_data(self, num, title, insights):
        self.add_page()
        self.chapter_title(num, title)
        self.set_font('Arial', '', 12)
        # Output justified text
        self.cell(0, 5, "Initial data plot")
        self.image(insights["initial_data_plot"], x=0, y=50, w=200, h=50, )
        self.ln()
        self.cell(0, 105, "Moving Average with window size of 5 applied to month granularity")
        self.image("report/moving_average_5_raw_to_month.png", x=0, y=110, w=200, h=50)
        self.ln()
        self.cell(0, 120, "Moving Average with window size of 30 applied to month granularity")
        self.image("report/moving_average_30_raw_to_month.png", x=0, y=150, w=200, h=50)
        self.ln()
        self.cell(0, 130, "Moving Average with window size of 90 applied to month granularity")
        self.image("report/moving_average_90_raw_to_month.png", x=0, y=230, w=200, h=50)
        self.ln()
