#include "BotClientSockets.h"
#include "BotTrackingServiceSocket.h"
#include "RemoteControllerSocket.h"
#include "CollisionDetector.h"
#include "GameRunner.h"
#include "VisualizationSocket.h"

#include <QCoreApplication>
#include <QJsonDocument>
#include <QCommandLineParser>

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    QCoreApplication::setApplicationName("GameService");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addVersionOption();
    QCommandLineOption portOffsetOption(QStringList() << "p" << "port-offset",
            QCoreApplication::translate("main", "Add <offset> to port numbers."),
            QCoreApplication::translate("main", "<offset>"),
            "0");
    parser.addOption(portOffsetOption);
    parser.process(app);

    int portOffset = parser.value(portOffsetOption).toInt();

    TeamSettings teamSettings;
    BotTrackingServiceSocket botTrackingServiceSocket(teamSettings);
    BotClientSockets botClientSockets;
    RemoteControllerSocket remoteControllerSocket(teamSettings);
    VisualizationSocket visualizationSocket;

    if (botTrackingServiceSocket.listen(QHostAddress::Any, static_cast<quint16>(9635 + portOffset)))
    {
        qDebug() << "BotTrackingServiceSocket listening";
    }
    else
    {
        qDebug() << "BotTrackingServiceSocket not listening";
    }

    if (botClientSockets.listen(QHostAddress::Any, static_cast<quint16>(9735 + portOffset)))
    {
        qDebug() << "BotClientSockets listening";
    }
    else
    {
        qDebug() << "BotClientSockets not listening";
    }
    if (visualizationSocket.listen(QHostAddress::Any, static_cast<quint16>(9835 + portOffset)))
    {
        qDebug() << "VisualizationSocket listening";
    }
    else
    {
        qDebug() << "VisualizationSocket not listening";
    }
    if (remoteControllerSocket.listen(QHostAddress::Any, static_cast<quint16>(9935 + portOffset)))
    {
        qDebug() << "RemoteControllerSocket listening";
    }
    else
    {
        qDebug() << "RemoteController not listening";
    }

    GameRunner gameRunner;

    QObject::connect(&gameRunner, &GameRunner::sendOutRevealedState, [&visualizationSocket] (const QJsonObject& state)
    {
        //qDebug().noquote().nospace() << "Sending out gameState" << state;
        visualizationSocket.sendState(state);
    });
    QObject::connect(&gameRunner, &GameRunner::sendOutObscuredState, [&botClientSockets] (const QJsonObject& state)
    {
        botClientSockets.sendState(state);
    });
    QObject::connect(& botTrackingServiceSocket, &BotTrackingServiceSocket::newBotLocations, [&gameRunner] (const QVector<BotInfo>& botLocations)
    {
        gameRunner.setBotLocations(botLocations);
    });
    QObject::connect(&remoteControllerSocket, &RemoteControllerSocket::createGame, [&gameRunner] (const GameOptions& gameOptions)
    {
        gameRunner.createNewGame(gameOptions);
    });
    QObject::connect(&remoteControllerSocket, &RemoteControllerSocket::startGame, [&gameRunner]
    {
        gameRunner.startGame();
    });
    QObject::connect(&remoteControllerSocket, &RemoteControllerSocket::stopGame, [&gameRunner]
    {
        gameRunner.stopGame();
    });

    return app.exec();
}
