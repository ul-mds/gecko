# Citing Gecko

If you find Gecko useful, and you wish to publish your work or research based on it, then we would love to see you
properly cite Gecko. 
This page is intended to point you to all our published research to date and to help you correctly cite it.

## Publications

### SoftwareX (Aug 10, 2024)

- Article: [Gecko: A Python library for the generation and mutation of realistic personal identification data at scale](https://www.sciencedirect.com/science/article/pii/S2352711024002176?via%3Dihub)
- DOI: [10.1016/j.softx.2024.101846](http://doi.org/10.1016/j.softx.2024.101846)

??? Abstract

    Record linkage algorithms require testing on realistic personal identification data to assess their efficacy in real-world settings. Access to this kind of data is often infeasible due to rigid data privacy regulations. Open-source tools for generating realistic data are either unmaintained or lack performance to scale to the generation of millions of records. We introduce Gecko as a Python library for creating shareable scripts to generate and mutate realistic personal data. Built on top of popular data science libraries in Python, it greatly facilitates integration into existing workflows. Benchmarks are provided to prove the library’s performance and scalability claims.

=== "APA"

    Jugl, M., & Kirsten, T. (2024). Gecko: A Python library for the generation and mutation of realistic personal identification data at scale. _SoftwareX, 27_, 101846. https://doi.org/10.1016/j.softx.2024.101846

=== "IEEE"

    M. Jugl and T. Kirsten, “Gecko: A Python library for the generation and mutation of realistic personal identification data at scale,” _SoftwareX_, vol. 27, p. 101846, Sep. 2024, doi: 10.1016/j.softx.2024.101846.

=== "ISO 690"

    JUGL, Maximilian and KIRSTEN, Toralf, 2024. Gecko: A Python library for the generation and mutation of realistic personal identification data at scale. _SoftwareX_. Vol. 27, p. 101846. DOI 10.1016/j.softx.2024.101846. 

=== "MLA"

    Jugl, Maximilian, and Toralf Kirsten. “Gecko: A Python Library for the Generation and Mutation of Realistic Personal Identification Data at Scale.” _SoftwareX_, vol. 27, Sept. 2024, p. 101846. _DOI.org (Crossref)_, https://doi.org/10.1016/j.softx.2024.101846.

=== "BibTeX"

    ```tex
    @article{jugl_gecko_2024,
        title = {Gecko: {A} {Python} library for the generation and mutation of realistic personal identification data at scale},
        volume = {27},
        issn = {2352-7110},
        shorttitle = {Gecko},
        url = {https://www.sciencedirect.com/science/article/pii/S2352711024002176},
        doi = {10.1016/j.softx.2024.101846},
        abstract = {Record linkage algorithms require testing on realistic personal identification data to assess their efficacy in real-world settings. Access to this kind of data is often infeasible due to rigid data privacy regulations. Open-source tools for generating realistic data are either unmaintained or lack performance to scale to the generation of millions of records. We introduce Gecko as a Python library for creating shareable scripts to generate and mutate realistic personal data. Built on top of popular data science libraries in Python, it greatly facilitates integration into existing workflows. Benchmarks are provided to prove the library’s performance and scalability claims.},
        urldate = {2024-10-17},
        journal = {SoftwareX},
        author = {Jugl, Maximilian and Kirsten, Toralf},
        month = sep,
        year = {2024},
        keywords = {Data privacy, Record linkage, Python, Data science},
        pages = {101846},
    }
    ```


## Presentations

### 18th Leipzig Research Festival for Life Sciences (Jan 30, 2025)

- Abstract: Improving record linkage quality on identification data in the Leipzig Obesity BioBank
- DOI: _to be released_
- Slides: [Download (PDF)](static/2025-01-30-Gecko-LOBB-LRF25.pdf)

??? Abstract

    Within longitudinal medical studies, identification data (IDAT) is collected from participants which enables them to be reidentified across multiple sessions. Trusted third parties may then consolidate medical records obtained from the same person before handing them off to data scientists to conduct their research. This process of merging records using stable or quasi-identifiers is referred to as “record linkage”. Due to the nature of these studies, errors arise in the collected data due to changing data entry methods and staff over time. Determining the required level of similarity to classify a record pair as a match therefore becomes a reoccurring challenge in the field of record linkage. 

    We present a method for estimating similarity thresholds for privacy-preserving record linkage (PPRL) using IDAT from two sub-studies of the Leipzig Obesity BioBank (LOBB). LOBB collects samples from over 8000 patients to conduct research on diseases related to obesity. We analyze the types and frequencies of typographical errors present in their IDAT. This information is used to infer a configuration for Gecko, which is a software library used to generate a set of realistic personal IDAT that closely replicates the errors found in the LOBB data. We then evaluate the impact of each error class on match quality using these datasets. Our findings provide needed insights into the challenges of integrating real-world IDAT, which is necessary in PPRL where access to such data is often prohibited, and underline its significance in enhancing research validity and reliability. 

### 69th Annual GMDS Conference (Sep 9, 2024)

- Abstract: [Generation and mutation of realistic personal identification data for the evaluation of record linkage algorithms](https://www.egms.de/static/de/meetings/gmds2024/24gmds020.shtml)
- DOI: [10.3205/24gmds020](https://dx.doi.org/10.3205/24gmds020)
- Slides: [Download (PDF)](static/2024-09-09-Gecko-GMDS.pptx.pdf)

??? Abstract

    Personal data is often scattered across various stakeholders due to its collection for various data collection purposes. This leads to a high degree of fragmentation, which necessitates the consolidation of multiple data sources in order to obtain a complete view of natural persons. Linking personal data records together is trivial with a globally unique personal identifier, but such an identifier is often either not available or out of scope in most scenarios. Algorithms from the field of record linkage have therefore been employed instead. They operate on identification data and assign a similarity to record pairs in order to decide whether they should be merged or not.

    These record linkage algorithms require testing on realistic data to evaluate their efficacy in real-world situations. However due to the sensitive nature of identification data, access to real-world testing data has been mostly exclusive to researchers with personal ties to medical institutions in the past. This has led to the creation of tools which generate personal data that seems realistic based on publicly available data sources. To the best of our knowledge, all previously published tools are either inactive, unmaintained, closed source or outdated.

    We present Gecko: an open-source Python library for the generation and mutation of personal identification data based on public data and error sources. It takes after GeCo which showed the promise of creating reproducible and shareable scripts to generate data. The ease of integration into data science applications of the original library leaves a lot to be desired. Gecko addresses this by reimplementing GeCo’s core features on top of popular data science libraries and extending them by fixing GeCo’s limitations, allowing the generation of arbitrarily complex multivariate data, fine-grain control over its randomized routines and data mutation across multiple instead of single fields.

    Gecko makes extensive use of Pandas data frames which allow exports of generated data in various interoperable file formats such as CSV. We validated that data generated by Gecko can be imported into E-PIX, which ensures Gecko’s compatibility with other tools with CSV parsing capabilities. Furthermore, we extensively benchmarked Gecko to ensure that it fulfills its performance claims. Despite the lack of test data from other solutions in the field, we estimate that Gecko’s single-core performance stands as best-in-class by a comfortable margin.

    Gecko’s performance and configurability allows it to generate datasets with millions of records for the validation of record linkage algorithms in reasonable time frames. Its capabilities to quickly generate data on-the-fly opens it up for use in other data science applications where realistic identification data may be needed. A publicly available data repository allows for quick testing of Gecko’s capabilities for library users. We encourage users of Gecko to donate data samples from various regions and languages in order to obtain higher multilingual coverage. Future versions of Gecko aim at providing export facilities for FHIR, as well as support for more complex error classes such as temporal errors and column shifts in data.

### 8th Freiberg PhD Conference (Jun 7, 2024)

- Abstract: [Gecko: generation and mutation of realistic identification data at scale for record linkage evaluation](https://nbn-resolving.org/urn:nbn:de:bsz:105-qucosa2-926058)
- URN: [urn:nbn:de:bsz:105-qucosa2-926058](https://nbn-resolving.org/urn:nbn:de:bsz:105-qucosa2-926058)
- Slides: [Download (PDF)](static/2024-06-07-Gecko.pptx.pdf)

??? Abstract

    Collection of personal data at different stakeholders for various purposes leads to fragmentation. Consolidation of these data sources is therefore necessary to get a complete view. In the absence of a globally unique personal identifier, record linkage algorithms are being used instead. These algorithms require testing on realistic data which, due to their sensitive nature, led to the development of data generation tools.

    We present Gecko: an open-source Python library for the generation and mutation of personal identification data at scale. Inspired by its predecessor GeCo, it provides its users with a set of functions to enable the creation of shareable and reproducible scripts for data generation. It comes with a simplified API and substantial performance gains due to the usage of popular scientific computing libraries under the hood. Aside from the validation of record linkage algorithms, its performance enables the integration into a variety of data science applications.