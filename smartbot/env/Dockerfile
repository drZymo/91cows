FROM alpine:latest
WORKDIR /root/
RUN apk --no-cache add make gcc g++ cmake git qt5-qtdeclarative-dev
RUN git clone https://github.com/drZymo/91cows
RUN mkdir build && cd build && cmake ../91cows/GameService && make

FROM alpine:latest
RUN apk --no-cache add qt5-qtbase-x11
COPY --from=0 /root/build/GameService/GameService .
ENTRYPOINT ["/GameService"]
